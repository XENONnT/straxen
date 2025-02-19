import os
import json
from warnings import warn
from typing import Dict

import numpy as np
import strax
from utilix import xent_collection

try:
    import admix
    from rucio.common.exception import DataIdentifierNotFound

    HAVE_ADMIX = True
except (ImportError, AttributeError):
    HAVE_ADMIX = False

export, __all__ = strax.exporter()

__all__.extend(["HAVE_ADMIX"])


@export
class TooMuchDataError(Exception):
    pass


@export
class RucioRemoteFrontend(strax.StorageFrontend):
    """Uses the rucio client for the data find."""

    storage_type = strax.StorageType.REMOTE
    local_did_cache = None
    path = None

    def __init__(
        self,
        download_heavy=False,
        staging_dir="./strax_data",
        rses_only=tuple(),
        tries=3,
        num_threads=1,
        stage=False,
        *args,
        **kwargs,
    ):
        """
        :param download_heavy: option to allow downloading of heavy data through RucioRemoteBackend
        :param args: Passed to strax.StorageFrontend
        :param kwargs: Passed to strax.StorageFrontend
        :param rses_only: tuple, limits RSE selection to these options if provided
        """
        super().__init__(*args, **kwargs)
        self.readonly = True
        self.collection = xent_collection()
        self.backends = []

        if HAVE_ADMIX:
            self.backends = [
                RucioRemoteBackend(
                    staging_dir,
                    rses_only=rses_only,
                    download_heavy=download_heavy,
                    tries=tries,
                    num_threads=num_threads,
                    stage=stage,
                ),
            ]
        else:
            self.log.warning(
                "You passed use_remote=True to rucio fronted, "
                "but you don't have access to admix/rucio! Using local backend only."
            )

    def find_several(self, keys, **kwargs):
        # for performance, dont do find_several with this storage frontend
        # we basically do the same query we would do in the RunDB plugin
        return np.zeros_like(keys, dtype=bool).tolist()

    def _find(self, key: strax.DataKey, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        did = key_to_rucio_did(key)
        if allow_incomplete or write:
            raise RuntimeError(
                "Allow incomplete/writing is not allowed for "
                f"{self.__class.__name__} since data might not be "
                "continuous"
            )
        try:
            for b in self.backends:
                rse = b._get_rse(did, state="OK")
                if rse:
                    return "RucioRemoteBackend", did
        except DataIdentifierNotFound:
            pass

        raise strax.DataNotAvailable

    def find(self, key: strax.DataKey, write=False, check_broken=False, **kwargs):
        # Overwrite defaults of super().find()
        return super().find(key, write, check_broken, **kwargs)


@export
class RucioRemoteBackend(strax.FileSytemBackend):
    """Get data from remote Rucio RSE."""

    # datatypes we don't want to download since they're too heavy
    heavy_types = ["raw_records", "raw_records_nv", "raw_records_he"]

    # for caching RSE locations
    dset_cache: Dict[str, str] = {}

    def __init__(
        self,
        staging_dir,
        download_heavy=False,
        rses_only=tuple(),
        tries=3,
        num_threads=1,
        stage=False,
        **kwargs,
    ):
        """
        :param staging_dir: Path (a string) where to save data. Must be
            a writable location.
        :param download_heavy: Whether or not to allow downloads of the
            heaviest data (raw_records*, less aqmon and MV)
        :param kwargs: Passed to strax.FileSystemBackend
        :param rses_only: tuple, limits RSE selection to these options if provided
        """
        mess = (
            f"You told the rucio backend to download data to {staging_dir}, "
            "but that path is not writable by your user"
        )
        if os.path.exists(staging_dir):
            if not os.access(staging_dir, os.W_OK):
                raise PermissionError(mess)
        else:
            try:
                os.makedirs(staging_dir)
            except OSError:
                raise PermissionError(mess)
        super().__init__(**kwargs)
        self.download_heavy = download_heavy
        self.staging_dir = staging_dir
        self.rses_only = strax.to_str_tuple(rses_only)
        self.tries = tries
        self.num_threads = num_threads
        self.stage = stage

    def _get_rse(self, dset_did, **filters):
        """Determine the appropriate Rucio Storage Element (RSE) for a dataset.

        :param dset_did (str) :The dataset identifier.
        :return (str) : The selected RSEs.
        ------
        Uses self.rses_only to filter available RSEs if set.

        """
        rses = admix.rucio.get_rses(dset_did, **filters)
        rses = list(set(rses) & set(self.rses_only)) if self.rses_only else rses
        rse = admix.downloader.determine_rse(rses)
        return rse

    def _get_metadata(self, dset_did, **kwargs):
        if dset_did in self.dset_cache:
            rse = self.dset_cache[dset_did]
        else:
            rse = self._get_rse(dset_did)
            self.dset_cache[dset_did] = rse

        metadata_did = strax.RUN_METADATA_PATTERN % dset_did
        warn(f"Downloading {metadata_did} from {rse}")
        downloaded = admix.download(
            metadata_did,
            location=self.staging_dir,
            rse=rse,
            tries=self.tries,
            num_threads=self.num_threads,
            stage=self.stage,
        )
        if len(downloaded) != 1:
            raise ValueError(f"{metadata_did} should be a single file. We found {len(downloaded)}.")
        metadata_path = downloaded[0]
        # check again
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found at {metadata_path}")

        with open(metadata_path, mode="r") as f:
            return json.loads(f.read())

    def _read_chunk(self, dset_did, chunk_info, dtype, compressor):
        base_dir = os.path.join(self.staging_dir, did_to_dirname(dset_did))
        chunk_file = chunk_info["filename"]
        chunk_path = os.path.abspath(os.path.join(base_dir, chunk_file))
        if not os.path.exists(chunk_path):
            number, datatype, hsh = parse_rucio_did(dset_did)
            if datatype in self.heavy_types and not self.download_heavy:
                error_msg = (
                    "For space reasons we don't want to have everyone "
                    "downloading raw data. If you know what you're "
                    "doing, pass download_heavy=True to the Rucio "
                    "frontend. If not, check your context and/or ask "
                    "someone if this raw data is needed locally."
                )
                warn(error_msg)
                raise strax.DataNotAvailable
            scope, name = dset_did.split(":")
            if dset_did in self.dset_cache:
                rse = self.dset_cache[dset_did]
            else:
                rse = self._get_rse(dset_did)
                self.dset_cache[dset_did] = rse

            chunk_did = f"{scope}:{chunk_file}"
            warn(f"Downloading {chunk_did} from {rse}")
            downloaded = admix.download(
                chunk_did, rse=rse, location=self.staging_dir, stage=self.stage
            )
            if len(downloaded) != 1:
                raise ValueError(
                    f"{chunk_did} should be a single file. We found {len(downloaded)}."
                )
            assert chunk_path == downloaded[0]

        # check again
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"No chunk file found at {chunk_path}")

        return strax.load_file(chunk_path, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata, **kwargs):
        raise NotImplementedError(
            "Cannot save directly into rucio (yet), upload with admix instead"
        )


@export
class RucioSaver(strax.Saver):
    """TODO: Saves data to rucio if you are the production user."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


@export
def parse_rucio_did(did: str) -> tuple:
    """Parses a Rucio DID and returns a tuple of (number:int, dtype:str, hash:str)"""
    scope, name = did.split(":")
    number = int(scope.split("_")[1])
    dtype, hsh = name.split("-")
    return number, dtype, hsh


def did_to_dirname(did: str):
    """Takes a Rucio dataset DID and returns a dirname like used by strax.FileSystemBackend."""
    # make sure it's a DATASET did, not e.g. a FILE
    if len(did.split("-")) != 2:
        raise RuntimeError(
            f"The DID {did} does not seem to be a dataset DID. "
            "Is it possible you passed a file DID?"
        )
    dirname = did.replace(":", "-").replace("xnt_", "")
    return dirname


@export
def key_to_rucio_did(key: strax.DataKey) -> str:
    """Convert a strax.datakey to a rucio did field in rundoc."""
    return f"xnt_{key.run_id}:{key.data_type}-{key.lineage_hash}"
