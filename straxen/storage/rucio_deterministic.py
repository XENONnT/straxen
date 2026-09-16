import hashlib
from typing import Optional, Tuple, Union
from urllib.parse import urlsplit
import strax

export, __all__ = strax.exporter()


@export
def rucio_deterministic_path(did: str, algorithm: str = "hash") -> str:
    """Convert a Rucio DID ('scope:filename') to relative path using Rucio deterministic convention.

    By default, uses Rucio's standard two-level MD5 directory hierarchy:
    <scope>/<md5[0:2]>/<md5[2:4]>/<filename>

    :param did: Rucio Data Identifier in format 'scope:filename'.
    :param algorithm: Hashing algorithm ('hash' or 'md5' for standard Rucio convention).
    :return: Relative path string (e.g. 'xnt_012345/a1/b2/records-5vbh5o52-000000').

    """
    if ":" not in did:
        raise ValueError(f"Invalid Rucio DID '{did}'. Expected 'scope:filename' format.")
    scope, filename = did.split(":", 1)

    if algorithm in ("hash", "md5"):
        # Rucio standard deterministic hashing: MD5 of 'scope:filename'
        md5_hex = hashlib.md5(did.encode("utf-8")).hexdigest()  # nosec
        t1 = md5_hex[0:2]
        t2 = md5_hex[2:4]
        return f"{scope}/{t1}/{t2}/{filename}"
    elif algorithm == "prefix":
        # Prefix algorithm: first two and next two chars of filename
        clean_fn = filename.lstrip("/")
        t1 = clean_fn[0:2] if len(clean_fn) >= 2 else "00"
        t2 = clean_fn[2:4] if len(clean_fn) >= 4 else "00"
        return f"{scope}/{t1}/{t2}/{filename}"
    else:
        raise ValueError(f"Unsupported Rucio deterministic algorithm: {algorithm}")


@export
def key_to_rucio_dids(
    key: Union[strax.DataKey, str], scope_prefix: str = "xnt_"
) -> Tuple[str, str]:
    """Convert a strax.DataKey or key string into (dataset_did, metadata_did).

    :param key: strax.DataKey instance or key string.
    :param scope_prefix: Scope prefix string (default 'xnt_').
    :return: Tuple of (dataset_did, metadata_did), e.g. ('xnt_012345:records-5vbh5o52',
        'xnt_012345:records-5vbh5o52-metadata.json')

    """
    if isinstance(key, strax.DataKey):
        run_id = str(key.run_id)
        data_type = str(key.data_type)
        lineage_hash = str(key.lineage_hash)
    else:
        # String representation, e.g. '012345-records-5vbh5o52' or 'xnt_012345:records-5vbh5o52'
        # Follows strax.DataKey.__str__ format: <run_id>-<data_type>-<lineage_hash>.
        # Note: Strax conventions enforce alphanumeric and underscore data type names
        # (e.g. 'raw_records', 'event_info'); data types containing hyphens are not supported
        # in hyphen-delimited string keys. Passing a strax.DataKey instance directly avoids
        # string parsing entirely.
        str_key = str(key).strip("/")
        if ":" in str_key:
            scope, name = str_key.split(":", 1)
            dataset_did = f"{scope}:{name}"
            md_name = strax.RUN_METADATA_PATTERN % name
            metadata_did = f"{scope}:{md_name}"
            return dataset_did, metadata_did
        parts = str_key.split("-")
        if len(parts) >= 3:
            run_id = parts[0]
            data_type = parts[1]
            lineage_hash = "-".join(parts[2:])
        else:
            raise ValueError(f"Cannot parse key '{key}' into run_id, data_type, and lineage_hash.")

    scope = f"{scope_prefix}{run_id}" if scope_prefix else run_id
    dataset_name = f"{data_type}-{lineage_hash}"
    dataset_did = f"{scope}:{dataset_name}"
    metadata_name = strax.RUN_METADATA_PATTERN % dataset_name
    metadata_did = f"{scope}:{metadata_name}"
    return dataset_did, metadata_did


@export
def chunk_to_rucio_did(scope: str, chunk_filename: str) -> str:
    """Format a chunk filename as a Rucio DID.

    :param scope: Rucio scope string (e.g. 'xnt_012345').
    :param chunk_filename: Chunk filename (e.g. 'records-5vbh5o52-000000').
    :return: Rucio DID string (e.g. 'xnt_012345:records-5vbh5o52-000000').

    """
    return f"{scope}:{chunk_filename}"


@export
def did_from_backend_key(backend_key: str) -> Optional[str]:
    """Extract Rucio DID from backend_key if present.

    Inspects the final path segment of a URL or file path for ':' indicating 'scope:name'.

    """
    parsed = urlsplit(backend_key)
    path = parsed.path.rstrip("/")
    if not path:
        return None
    last_segment = path.split("/")[-1]
    if ":" in last_segment:
        return last_segment
    return None
