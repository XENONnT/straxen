from collections import OrderedDict
from dataclasses import dataclass
import errno
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit
import strax

from .rucio_deterministic import (
    chunk_to_rucio_did,
    key_to_rucio_dids,
    rucio_deterministic_path,
)

log = logging.getLogger("straxen.storage.xrootd")

try:
    import utilix
except (ImportError, RuntimeError, FileNotFoundError):
    utilix = None

try:
    import fsspec

    HAVE_FSSPEC = True
except ImportError:
    HAVE_FSSPEC = False

DEFAULT_REDIRECTOR_URL = "root://midway-origin.uchicago.edu/"
DEFAULT_SUBPATH = "xenon/xenonnt/processed"

export, __all__ = strax.exporter()
__all__.extend(
    [
        "HAVE_FSSPEC",
        "DEFAULT_REDIRECTOR_URL",
        "DEFAULT_SUBPATH",
        "normalize_redirector_url",
        "parse_redirector_list",
        "RedirectorPool",
        "is_endpoint_failure",
    ]
)


@export
def is_endpoint_failure(exc: Optional[Exception]) -> bool:
    """Determine whether an exception indicates an endpoint outage or auth failure rather than
    missing data."""
    if exc is None:
        return False
    if isinstance(exc, FileNotFoundError):
        return False
    if hasattr(exc, "errno") and exc.errno == errno.ENOENT:
        return False
    msg = str(exc).lower()
    if "no such file" in msg or "not found" in msg or "[3011]" in msg:
        return False
    return True


def _resolve_uconfig(uconfig: Optional[Any] = None) -> Optional[Any]:
    """Resolve utilix configuration instance if available."""
    if uconfig is not None:
        return uconfig
    if utilix is not None and getattr(utilix, "uconfig", None) is not None:
        return utilix.uconfig
    try:
        import straxen

        return getattr(straxen, "uconfig", None)
    except ImportError:
        return None


@export
def normalize_redirector_url(url: str) -> str:
    """Normalize a redirector URL ensuring clean protocol and no trailing slash."""
    url = str(url).strip()
    if not url:
        return ""
    if "://" not in url:
        url = f"root://{url}"
    protocol, rest = url.split("://", 1)
    rest_clean = rest.rstrip("/")
    return f"{protocol}://{rest_clean}" if rest_clean else f"{protocol}://"


@export
def parse_redirector_list(redirectors: Union[str, List[str], Tuple[str, ...], None]) -> List[str]:
    """Parse a list, tuple, or comma/space-delimited string into normalized redirector URLs."""
    if redirectors is None:
        return []
    if isinstance(redirectors, str):
        raw_items = [
            item.strip() for part in redirectors.split(",") for item in part.split() if item.strip()
        ]
    else:
        raw_items = [str(r).strip() for r in redirectors if str(r).strip()]

    normalized: List[str] = []
    for item in raw_items:
        norm = normalize_redirector_url(item)
        if norm and norm not in normalized:
            normalized.append(norm)
    return normalized


@export
class RedirectorPool:
    """Manages an ordered pool of XRootD redirector endpoints with dynamic failover tracking."""

    def __init__(
        self,
        redirectors: Union[str, List[str], Tuple[str, ...], None],
        failover_policy: str = "priority",
        cooldown_seconds: float = 300.0,
    ):
        """
        :param redirectors: Candidate redirector URLs (string, list, tuple, or comma-delimited).
        :param failover_policy: Failover routing policy ('priority' or 'round_robin').
        :param cooldown_seconds: Cooldown interval before retrying a failed redirector.
        """
        self.raw_redirectors = parse_redirector_list(redirectors)
        if not self.raw_redirectors:
            self.raw_redirectors = [normalize_redirector_url(DEFAULT_REDIRECTOR_URL)]

        self.failover_policy = failover_policy
        self.cooldown_seconds = cooldown_seconds

        self._active_index = 0
        self._failure_counts: Dict[str, int] = {r: 0 for r in self.raw_redirectors}
        self._last_failure_time: Dict[str, float] = {r: 0.0 for r in self.raw_redirectors}
        self._round_robin_counter = 0

    @property
    def redirectors(self) -> List[str]:
        return list(self.raw_redirectors)

    @property
    def active_redirector(self) -> str:
        if not self.raw_redirectors:
            return normalize_redirector_url(DEFAULT_REDIRECTOR_URL)
        return self.raw_redirectors[self._active_index % len(self.raw_redirectors)]

    @active_redirector.setter
    def active_redirector(self, redirector: str) -> None:
        norm = normalize_redirector_url(redirector)
        if norm in self.raw_redirectors:
            self._active_index = self.raw_redirectors.index(norm)
        else:
            self.raw_redirectors.append(norm)
            self._active_index = len(self.raw_redirectors) - 1
            self._failure_counts[norm] = 0
            self._last_failure_time[norm] = 0.0

    def mark_success(self, redirector: str) -> None:
        """Mark a redirector as successful and set it as active."""
        norm = normalize_redirector_url(redirector)
        self.active_redirector = norm
        self._failure_counts[norm] = 0

    def mark_failure(self, redirector: str, error: Optional[Exception] = None) -> None:
        """Record a failure for the specified redirector and advance to the next candidate."""
        norm = normalize_redirector_url(redirector)
        self._failure_counts[norm] = self._failure_counts.get(norm, 0) + 1
        self._last_failure_time[norm] = time.time()
        log.warning(f"Redirector {norm} failed (count: {self._failure_counts[norm]}): {error}")
        if norm == self.active_redirector and len(self.raw_redirectors) > 1:
            candidates = [r for r in self.raw_redirectors if r != norm]
            if candidates:
                candidates.sort(key=lambda r: self._failure_counts.get(r, 0))
                self.active_redirector = candidates[0]

    def get_candidates(self) -> List[str]:
        """Return candidate redirectors in order of priority or round-robin."""
        if not self.raw_redirectors:
            return [normalize_redirector_url(DEFAULT_REDIRECTOR_URL)]
        if len(self.raw_redirectors) == 1:
            return list(self.raw_redirectors)

        now = time.time()
        available = []
        in_cooldown = []
        for r in self.raw_redirectors:
            last_fail = self._last_failure_time.get(r, 0.0)
            if self._failure_counts.get(r, 0) > 0 and (now - last_fail) < self.cooldown_seconds:
                in_cooldown.append(r)
            else:
                available.append(r)

        pool = available if available else list(self.raw_redirectors)

        if self.failover_policy == "round_robin":
            n = len(pool)
            start = self._round_robin_counter % n
            self._round_robin_counter += 1
            ordered = pool[start:] + pool[:start]
        else:  # priority
            active = self.active_redirector
            others = [r for r in pool if r != active]
            others.sort(
                key=lambda r: (
                    self._failure_counts.get(r, 0),
                    self.raw_redirectors.index(r),
                )
            )
            ordered = ([active] if active in pool else []) + others

        for r in in_cooldown:
            if r not in ordered:
                ordered.append(r)
        return ordered

    def swap_redirector_in_url(self, url: str, new_redirector: str) -> str:
        """Swap redirector origin in url while strictly preserving double slash '//' and subpath."""
        norm_new = normalize_redirector_url(new_redirector)
        old_parsed = urlsplit(url)
        new_parsed = urlsplit(norm_new)

        new_scheme = new_parsed.scheme or old_parsed.scheme or "root"
        new_netloc = new_parsed.netloc or (
            new_parsed.path.strip("/") if not new_parsed.scheme else ""
        )

        # If it is a file:// URL without netloc
        if new_scheme == "file" or old_parsed.scheme == "file":
            matched_old = None
            for r in self.raw_redirectors:
                r_norm = normalize_redirector_url(r).rstrip("/")
                if url == r_norm or url.startswith(r_norm + "/"):
                    matched_old = r_norm
                    break
            if matched_old:
                rel = url[len(matched_old) :].lstrip("/")
                return f"{norm_new.rstrip('/')}/{rel}"
            return url

        # Networked and virtual schemes (root://, memory://, http://)
        if old_parsed.netloc and new_netloc:
            if old_parsed.netloc == new_netloc and old_parsed.scheme == new_scheme:
                return url
            is_double_slash = old_parsed.path.startswith("//")
            path_remainder = old_parsed.path.lstrip("/")
            if is_double_slash or new_scheme == "root":
                return f"{new_scheme}://{new_netloc}//{path_remainder}"
            else:
                return f"{new_scheme}://{new_netloc}/{path_remainder}"

        return url


@export
@dataclass
class SciTokenInfo:
    """Discovered SciToken / WLCG Bearer Token authentication information."""

    token: Optional[str] = None
    token_file: Optional[str] = None
    source: str = "none"

    @property
    def is_valid(self) -> bool:
        return bool(self.token or self.token_file)

    def read_token(self) -> Optional[str]:
        """Read and return the token string from token_file or token attribute."""
        if self.token:
            return self.token
        if self.token_file and os.path.isfile(self.token_file):
            try:
                with open(self.token_file, "r") as f:
                    return f.read().strip()
            except OSError:
                return None
        return None


@export
def discover_scitoken(
    explicit_token: Optional[str] = None,
    explicit_token_file: Optional[str] = None,
    uconfig: Optional[Any] = None,
    sync_environ: bool = True,
) -> SciTokenInfo:
    """Discover SciToken / WLCG Bearer Token credentials across standard environments.

    Search precedence:
    1. Explicit arguments (explicit_token_file, explicit_token)
    2. Environment variables (BEARER_TOKEN, BEARER_TOKEN_FILE, SCITOKEN, SCITOKEN_FILE)
    3. utilix.uconfig [xrootd] / [scitoken] sections (token_file, token)
    4. WLCG standard paths ($XDG_RUNTIME_DIR/bt_u<uid>, /tmp/bt_u<uid>, /tmp/scitoken_u<uid>)
    5. HTCondor scratch / job paths ($_CONDOR_JOB_IWD/.condor/tokens.d/, etc.)

    :param explicit_token: Explicit bearer token string.
    :param explicit_token_file: Explicit path to bearer token file.
    :param uconfig: Optional ConfigParser-like object (defaults to utilix.uconfig).
    :param sync_environ: If True and a token is discovered, exports BEARER_TOKEN_FILE
        or BEARER_TOKEN to os.environ for underlying C++ XRootD clients.
    :return: SciTokenInfo object describing discovered token and origin.

    """
    token_info: Optional[SciTokenInfo] = None

    # 1. Explicit arguments
    if explicit_token_file:
        if os.path.isfile(explicit_token_file) and os.path.getsize(explicit_token_file) > 0:
            token_info = SciTokenInfo(
                token_file=os.path.abspath(explicit_token_file),
                source="explicit:token_file",
            )
        else:
            log.warning(f"Explicit token_file does not exist or is empty: {explicit_token_file}")

    if token_info is None and explicit_token:
        stripped = explicit_token.strip()
        if stripped:
            token_info = SciTokenInfo(
                token=stripped,
                source="explicit:token",
            )

    # 2. Environment variables (WLCG precedence: BEARER_TOKEN before BEARER_TOKEN_FILE)
    if token_info is None and "BEARER_TOKEN" in os.environ and os.environ["BEARER_TOKEN"].strip():
        token_info = SciTokenInfo(
            token=os.environ["BEARER_TOKEN"].strip(),
            source="env:BEARER_TOKEN",
        )
    if token_info is None and "BEARER_TOKEN_FILE" in os.environ:
        bt_file = os.environ["BEARER_TOKEN_FILE"]
        if os.path.isfile(bt_file) and os.path.getsize(bt_file) > 0:
            token_info = SciTokenInfo(
                token_file=os.path.abspath(bt_file),
                source="env:BEARER_TOKEN_FILE",
            )
    if token_info is None and "SCITOKEN" in os.environ and os.environ["SCITOKEN"].strip():
        token_info = SciTokenInfo(
            token=os.environ["SCITOKEN"].strip(),
            source="env:SCITOKEN",
        )
    if token_info is None and "SCITOKEN_FILE" in os.environ:
        st_file = os.environ["SCITOKEN_FILE"]
        if os.path.isfile(st_file) and os.path.getsize(st_file) > 0:
            token_info = SciTokenInfo(
                token_file=os.path.abspath(st_file),
                source="env:SCITOKEN_FILE",
            )

    # 3. utilix.uconfig
    if token_info is None:
        cfg = _resolve_uconfig(uconfig)
        if cfg is not None:
            for section in ("xrootd", "scitoken", "token"):
                if cfg.has_section(section):
                    # Check token_file keys first
                    for k in ("token_file", "bearer_token_file", "scitoken_file"):
                        val = cfg.get(section, k, fallback=None)
                        if val and os.path.isfile(val) and os.path.getsize(val) > 0:
                            token_info = SciTokenInfo(
                                token_file=os.path.abspath(val),
                                source=f"uconfig:{section}:{k}",
                            )
                            break
                    if token_info is not None:
                        break

                    # Check token string keys
                    for k in ("token", "bearer_token", "scitoken"):
                        val = cfg.get(section, k, fallback=None)
                        if val and val.strip():
                            stripped = val.strip()
                            if os.path.isfile(stripped) and os.path.getsize(stripped) > 0:
                                token_info = SciTokenInfo(
                                    token_file=os.path.abspath(stripped),
                                    source=f"uconfig:{section}:{k}",
                                )
                            else:
                                token_info = SciTokenInfo(
                                    token=stripped,
                                    source=f"uconfig:{section}:{k}",
                                )
                            break
                    if token_info is not None:
                        break

    # 4. WLCG standard paths
    if token_info is None:
        uid = os.getuid() if hasattr(os, "getuid") else None
        if uid is not None:
            xdg_dir = os.environ.get("XDG_RUNTIME_DIR")
            if xdg_dir:
                candidate = os.path.join(xdg_dir, f"bt_u{uid}")
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    token_info = SciTokenInfo(
                        token_file=candidate,
                        source="wlcg:xdg_runtime_dir",
                    )

            if token_info is None:
                candidate = f"/tmp/bt_u{uid}"
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    token_info = SciTokenInfo(
                        token_file=candidate,
                        source="wlcg:tmp",
                    )

            if token_info is None:
                candidate = f"/tmp/scitoken_u{uid}"
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    token_info = SciTokenInfo(
                        token_file=candidate,
                        source="osg:tmp",
                    )

    # 5. HTCondor tokens
    if token_info is None:
        for condor_var in ("_CONDOR_JOB_IWD", "_CONDOR_SCRATCH_DIR"):
            condor_base = os.environ.get(condor_var)
            if condor_base:
                tokens_dir = os.path.join(condor_base, ".condor", "tokens.d")
                if os.path.isdir(tokens_dir):
                    try:
                        for fname in sorted(os.listdir(tokens_dir)):
                            candidate = os.path.join(tokens_dir, fname)
                            if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                                token_info = SciTokenInfo(
                                    token_file=candidate,
                                    source=f"condor:{condor_var}",
                                )
                                break
                    except OSError:
                        pass
            if token_info is not None:
                break

    if token_info is None:
        token_info = SciTokenInfo(source="none")

    if sync_environ and token_info.is_valid:
        if token_info.token_file:
            os.environ["BEARER_TOKEN_FILE"] = token_info.token_file
            if token_info.source != "env:BEARER_TOKEN":
                os.environ.pop("BEARER_TOKEN", None)
        elif token_info.token:
            os.environ["BEARER_TOKEN"] = token_info.token
            if token_info.source != "env:BEARER_TOKEN_FILE":
                os.environ.pop("BEARER_TOKEN_FILE", None)

    return token_info


@export
class XRootDBackend(strax.StorageBackend):
    """Storage backend streaming chunks directly over root:// using fsspec and fsspec-xrootd.

    Supports both standard Strax hierarchical directory layouts and Rucio deterministic two-level
    MD5 hashing layouts with automated multi-redirector failover.

    """

    def __init__(
        self,
        xrootd_kwargs: Optional[dict] = None,
        set_target_chunk_mb: Optional[int] = None,
        cache_metadata: bool = True,
        max_cache_size: int = 1000,
        rucio_mode: bool = False,
        scope_prefix: str = "xnt_",
        redirector_pool: Optional[RedirectorPool] = None,
        *args,
        **kwargs,
    ):
        """
        :param xrootd_kwargs: Keyword arguments passed to fsspec.core.url_to_fs.
        :param set_target_chunk_mb: Target chunk size override in MB.
        :param cache_metadata: If True, cache parsed metadata in memory.
        :param max_cache_size: Maximum entries in bounded metadata LRU cache (default 1000).
        :param rucio_mode: If True, use Rucio deterministic MD5 hashing paths.
        :param scope_prefix: Scope prefix string used for Rucio DIDs (default 'xnt_').
        :param redirector_pool: Optional RedirectorPool for dynamic failover.
        """
        if not HAVE_FSSPEC:
            raise ImportError(
                "fsspec is required to use XRootDBackend. "
                "Please install it via 'pip install fsspec fsspec-xrootd' or conda."
            )
        super().__init__(*args, **kwargs)
        self.xrootd_kwargs = dict(xrootd_kwargs or {})
        self.set_chunk_size_mb = set_target_chunk_mb
        self.cache_metadata = cache_metadata
        self.max_cache_size = max_cache_size
        self.rucio_mode = rucio_mode
        self.scope_prefix = scope_prefix
        self.redirector_pool = redirector_pool
        self._metadata_cache: OrderedDict[str, dict] = OrderedDict()

    def clear_metadata_cache(self) -> None:
        """Clear the in-memory metadata cache."""
        self._metadata_cache.clear()

    def _get_fs_and_path(self, backend_key: str):
        """Extract filesystem instance and physical path for a backend key."""
        return fsspec.core.url_to_fs(backend_key, **self.xrootd_kwargs)

    def _get_metadata(self, backend_key: Union[strax.DataKey, str], **kwargs) -> dict:
        """Retrieve and parse JSON metadata from remote or local storage.

        Supports standard Strax candidate files, Rucio deterministic hashing paths, and multi-
        redirector failover.

        """
        key_str = str(backend_key)
        if self.cache_metadata and key_str in self._metadata_cache:
            self._metadata_cache.move_to_end(key_str)
            return self._metadata_cache[key_str]

        is_rucio = self.rucio_mode or (":" in key_str.split("/")[-1])

        candidates = []
        if is_rucio:
            last_part = key_str.split("/")[-1]
            base_url = key_str.rsplit("/", 1)[0]
            if ":" in last_part:
                did = last_part
            else:
                did, _ = key_to_rucio_dids(last_part, scope_prefix=self.scope_prefix)

            if "-metadata.json" in did:
                md_did = did
            else:
                scope, name = did.split(":", 1)
                md_did = f"{scope}:{strax.RUN_METADATA_PATTERN % name}"

            rel_path = rucio_deterministic_path(md_did)
            candidates.append(f"{base_url}/{rel_path}")
        else:
            fs, path = self._get_fs_and_path(key_str)
            if path.endswith(".json"):
                candidates = [key_str]
            else:
                clean_path = key_str.rstrip("/")
                folder_name = clean_path.split("/")[-1].replace("_temp", "")
                if "-" in folder_name:
                    try:
                        prefix = folder_name.split("-", maxsplit=1)[1]
                        candidates.append(f"{clean_path}/{strax.RUN_METADATA_PATTERN % prefix}")
                    except IndexError:
                        pass
                candidates.append(f"{clean_path}/metadata.json")

        redirectors: Sequence[Optional[str]]
        if self.redirector_pool is not None:
            redirectors = self.redirector_pool.get_candidates()
        else:
            redirectors = [None]

        last_error = None
        for red in redirectors:
            for candidate in candidates:
                cand_url = (
                    self.redirector_pool.swap_redirector_in_url(candidate, red)
                    if (self.redirector_pool is not None and red is not None)
                    else candidate
                )
                try:
                    fs, path = self._get_fs_and_path(cand_url)
                    with fs.open(path, mode="rb") as f:
                        content = f.read()
                        md = json.loads(content.decode("utf-8"))
                        if self.cache_metadata:
                            if len(self._metadata_cache) >= self.max_cache_size:
                                self._metadata_cache.popitem(last=False)
                            self._metadata_cache[key_str] = md
                        if self.redirector_pool is not None and red is not None:
                            self.redirector_pool.mark_success(red)
                        return md
                except (
                    FileNotFoundError,
                    IOError,
                    OSError,
                    TimeoutError,
                    ConnectionError,
                ) as e:
                    last_error = e
                    continue
                except json.JSONDecodeError as e:
                    raise strax.DataCorrupted(
                        f"Corrupted JSON in metadata file {cand_url}: {e}"
                    ) from e
            if (
                self.redirector_pool is not None
                and red is not None
                and is_endpoint_failure(last_error)
            ):
                self.redirector_pool.mark_failure(red, last_error)

        raise strax.DataNotAvailable(
            f"No metadata found for {backend_key}. Tried: {candidates}. Last error: {last_error}"
        )

    def _read_chunk(self, backend_key: str, chunk_info: dict, dtype, compressor: str):
        """Stream a single chunk directly into memory with automatic failover."""
        key_str = str(backend_key)
        chunk_fn = chunk_info["filename"]
        is_rucio = self.rucio_mode or (":" in key_str.split("/")[-1])

        if is_rucio:
            last_part = key_str.split("/")[-1]
            base_url = key_str.rsplit("/", 1)[0]
            if ":" in last_part:
                did = last_part
            else:
                did, _ = key_to_rucio_dids(last_part, scope_prefix=self.scope_prefix)
            scope = did.split(":", 1)[0]
            chunk_did = chunk_to_rucio_did(scope, chunk_fn)
            rel_path = rucio_deterministic_path(chunk_did)
            primary_chunk_file = f"{base_url}/{rel_path}"
        else:
            primary_chunk_file = f"{key_str.rstrip('/')}/{chunk_fn}"

        redirectors: Sequence[Optional[str]]
        if self.redirector_pool is not None:
            redirectors = self.redirector_pool.get_candidates()
        else:
            redirectors = [None]

        last_error = None
        for red in redirectors:
            chunk_url = (
                self.redirector_pool.swap_redirector_in_url(primary_chunk_file, red)
                if (self.redirector_pool is not None and red is not None)
                else primary_chunk_file
            )
            try:
                fs, path = self._get_fs_and_path(chunk_url)
                with fs.open(path, mode="rb") as f:
                    data = strax.load_file(f, dtype=dtype, compressor=compressor)
                    if self.redirector_pool is not None and red is not None:
                        self.redirector_pool.mark_success(red)
                    return data
            except (
                FileNotFoundError,
                IOError,
                OSError,
                TimeoutError,
                ConnectionError,
            ) as e:
                last_error = e
                if self.redirector_pool is not None and red is not None and is_endpoint_failure(e):
                    self.redirector_pool.mark_failure(red, e)
                continue

        raise strax.DataNotAvailable(
            f"Chunk file not found or inaccessible at {primary_chunk_file}. "
            f"Last error: {last_error}"
        ) from last_error

    def _read_and_format_chunk(self, *args, **kwargs) -> strax.Chunk:
        """Format chunk and optionally apply target chunk size."""
        chunk = super()._read_and_format_chunk(*args, **kwargs)
        if self.set_chunk_size_mb:
            chunk.target_size_mb = self.set_chunk_size_mb
        return chunk

    def _saver(self, key, metadata, **kwargs):
        raise NotImplementedError("Writing via XRootD backend is not supported.")


@export
class XRootDFrontend(strax.StorageFrontend):
    """Storage frontend resolving data keys to remote root:// URIs for streaming.

    Supports dynamic utilix configuration, SciToken discovery, Rucio deterministic two-level
    hashing, and multi-redirector failover.

    """

    storage_type = strax.StorageType.REMOTE

    def __init__(
        self,
        redirector_url: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        redirector_urls: Optional[Union[List[str], Tuple[str, ...], str]] = None,
        fallback_redirectors: Optional[Union[List[str], Tuple[str, ...], str]] = None,
        subpath: Optional[str] = None,
        token: Optional[str] = None,
        token_file: Optional[str] = None,
        discover_token: bool = True,
        uconfig: Optional[Any] = None,
        xrootd_kwargs: Optional[dict] = None,
        set_target_chunk_mb: Optional[int] = None,
        cache_metadata: bool = True,
        max_cache_size: int = 1000,
        rucio_mode: bool = False,
        scope_prefix: str = "xnt_",
        failover_policy: str = "priority",
        *args,
        **kwargs,
    ):
        """
        :param redirector_url: Base XRootD redirector URL (or list of redirectors).
        :param redirector_urls: Primary list/string of candidate redirectors.
        :param fallback_redirectors: Optional list/string of fallback redirectors.
        :param subpath: Base subdirectory path on the server where runs are stored.
        :param token: Explicit bearer token string for authentication.
        :param token_file: Explicit path to bearer token file for authentication.
        :param discover_token: If True, automatically discover SciToken / WLCG Bearer Token.
        :param uconfig: Optional custom ConfigParser instance (defaults to utilix.uconfig).
        :param xrootd_kwargs: Optional dictionary of keyword arguments for fsspec/xrootd.
        :param set_target_chunk_mb: Override chunk size in MB returned by loader.
        :param cache_metadata: If True, cache parsed metadata in memory.
        :param max_cache_size: Maximum entries in bounded metadata LRU cache (default 1000).
        :param rucio_mode: If True, use Rucio deterministic MD5 hashing layout.
        :param scope_prefix: Scope prefix string for Rucio DIDs (default 'xnt_').
        :param failover_policy: Routing policy ('priority' or 'round_robin').
        """
        if not HAVE_FSSPEC:
            raise ImportError(
                "fsspec is required to use XRootDFrontend. "
                "Please install it via 'pip install fsspec fsspec-xrootd' or conda."
            )
        kwargs.setdefault("readonly", True)
        super().__init__(*args, **kwargs)

        self.uconfig = uconfig
        self.set_target_chunk_mb = set_target_chunk_mb
        self.cache_metadata = cache_metadata
        self.max_cache_size = max_cache_size
        self.raw_xrootd_kwargs = dict(xrootd_kwargs or {})

        self.init_redirector_url = redirector_url
        self.init_redirector_urls = redirector_urls
        self.init_fallback_redirectors = fallback_redirectors
        self.init_subpath = subpath
        self.init_token = token
        self.init_token_file = token_file
        self.discover_token = discover_token
        self.init_rucio_mode = rucio_mode
        self.scope_prefix = scope_prefix
        self.failover_policy = failover_policy

        self._configure()

    def _configure(self) -> None:
        """Resolve dynamic configuration from uconfig, environment, and auth tokens."""
        cfg = _resolve_uconfig(self.uconfig)

        # 1. Resolve redirector candidates
        candidates: List[str] = []

        if self.init_redirector_urls is not None:
            candidates.extend(parse_redirector_list(self.init_redirector_urls))
        elif self.init_redirector_url is not None:
            candidates.extend(parse_redirector_list(self.init_redirector_url))

        if self.init_fallback_redirectors is not None:
            candidates.extend(parse_redirector_list(self.init_fallback_redirectors))

        if not candidates and cfg is not None and cfg.has_section("xrootd"):
            u_urls = cfg.get("xrootd", "redirector_urls", fallback=None)
            if u_urls is None:
                u_urls = cfg.get("xrootd", "urls", fallback=None)
            if u_urls:
                candidates.extend(parse_redirector_list(u_urls))

            u_single = cfg.get("xrootd", "redirector_url", fallback=None)
            if u_single is None:
                u_single = cfg.get("xrootd", "url", fallback=None)
            if u_single:
                candidates.extend(parse_redirector_list(u_single))

            u_fallbacks = cfg.get("xrootd", "fallback_redirectors", fallback=None)
            if u_fallbacks is None:
                u_fallbacks = cfg.get("xrootd", "fallbacks", fallback=None)
            if u_fallbacks:
                candidates.extend(parse_redirector_list(u_fallbacks))

        if not candidates:
            candidates = [DEFAULT_REDIRECTOR_URL]

        self.redirector_pool = RedirectorPool(
            candidates,
            failover_policy=self.failover_policy,
        )
        self.redirector_url = self.redirector_pool.active_redirector

        # 2. Resolve subpath
        resolved_subpath = self.init_subpath
        if resolved_subpath is None:
            if cfg is not None and cfg.has_section("xrootd"):
                resolved_subpath = cfg.get("xrootd", "subpath", fallback=None)
            if resolved_subpath is None:
                resolved_subpath = DEFAULT_SUBPATH
        self.subpath = resolved_subpath.strip("/")

        # 3. Resolve rucio_mode
        self.rucio_mode = self.init_rucio_mode
        if not self.rucio_mode and cfg is not None and cfg.has_section("xrootd"):
            val = cfg.get("xrootd", "rucio_mode", fallback=None)
            if val is None:
                val = cfg.get("xrootd", "deterministic", fallback=None)
            if val is not None:
                self.rucio_mode = val.strip().lower() in ("true", "1", "yes")

        # 4. Resolve xrootd_kwargs and uconfig additions
        resolved_kwargs = dict(self.raw_xrootd_kwargs)
        if cfg is not None and cfg.has_section("xrootd"):
            timeout = cfg.get("xrootd", "timeout", fallback=None)
            if timeout is not None and "timeout" not in resolved_kwargs:
                try:
                    resolved_kwargs["timeout"] = int(float(timeout))
                except (ValueError, TypeError):
                    resolved_kwargs["timeout"] = timeout

        if "timeout" in resolved_kwargs:
            try:
                resolved_kwargs["timeout"] = int(float(resolved_kwargs["timeout"]))
            except (ValueError, TypeError):
                pass

        # 5. SciToken auth discovery
        if self.discover_token:
            self.token_info = discover_scitoken(
                explicit_token=self.init_token,
                explicit_token_file=self.init_token_file,
                uconfig=cfg,
                sync_environ=True,
            )
        else:
            self.token_info = SciTokenInfo(
                token=self.init_token,
                token_file=self.init_token_file,
                source="explicit" if (self.init_token or self.init_token_file) else "none",
            )
            if self.token_info.is_valid:
                if self.token_info.token_file:
                    os.environ["BEARER_TOKEN_FILE"] = self.token_info.token_file
                    os.environ.pop("BEARER_TOKEN", None)
                elif self.token_info.token:
                    os.environ["BEARER_TOKEN"] = self.token_info.token
                    os.environ.pop("BEARER_TOKEN_FILE", None)

        self.xrootd_kwargs = resolved_kwargs

        # 6. Initialize / reinitialize backend
        self.backends = [
            XRootDBackend(
                xrootd_kwargs=self.xrootd_kwargs,
                set_target_chunk_mb=self.set_target_chunk_mb,
                cache_metadata=self.cache_metadata,
                max_cache_size=self.max_cache_size,
                rucio_mode=self.rucio_mode,
                scope_prefix=self.scope_prefix,
                redirector_pool=self.redirector_pool,
            )
        ]

    def reload_config(self) -> None:
        """Dynamically reload configuration and credentials from utilix and environment."""
        self._configure()

    def clear_metadata_cache(self) -> None:
        """Clear metadata cache across all associated backends."""
        for backend in self.backends:
            if hasattr(backend, "clear_metadata_cache"):
                backend.clear_metadata_cache()

    def build_url(
        self,
        key: Union[strax.DataKey, str],
        redirector: Optional[str] = None,
    ) -> str:
        """Build authoritative remote physical file URL for the given key.

        Enforces the mandatory double slash '//' convention for root:// endpoints. Supports both
        standard Strax directory layouts and Rucio deterministic naming.

        """
        target_redirector = redirector or self.redirector_pool.active_redirector
        sub = self.subpath.strip("/")

        if self.rucio_mode:
            dataset_did, _ = key_to_rucio_dids(key, scope_prefix=self.scope_prefix)
            rel_path = f"{sub}/{dataset_did}" if sub else dataset_did
        else:
            rel_path = f"{sub}/{key}" if sub else str(key)
        rel_path = rel_path.strip("/")

        parsed = urlsplit(target_redirector)
        if parsed.scheme == "root":
            netloc = parsed.netloc or parsed.path.strip("/")
            base_path = parsed.path.strip("/") if parsed.netloc else ""
            full_rel = f"{base_path}/{rel_path}".strip("/") if base_path else rel_path
            return f"root://{netloc}//{full_rel}"
        elif parsed.scheme:
            if parsed.netloc:
                base_path = parsed.path.rstrip("/")
                return f"{parsed.scheme}://{parsed.netloc}{base_path}/{rel_path}"
            else:
                base_path = parsed.path.rstrip("/")
                return (
                    f"{parsed.scheme}://{base_path}/{rel_path}"
                    if base_path
                    else f"{parsed.scheme}:///{rel_path}"
                )
        else:
            base = target_redirector.rstrip("/")
            return f"{base}/{rel_path}"

    def _find(
        self,
        key: strax.DataKey,
        write: bool,
        allow_incomplete: bool,
        fuzzy_for: tuple,
        fuzzy_for_options: tuple,
    ):
        if write:
            raise strax.DataNotAvailable(f"{self} is strictly read-only.")

        backend = self.backends[0]
        # Start probing using the currently active redirector.
        # Backend.get_metadata() internally iterates through all candidates in the shared
        # RedirectorPool upon failure, avoiding redundant outer O(n^2) retries.
        initial_backend_key = self.build_url(key, redirector=self.redirector_pool.active_redirector)
        try:
            backend.get_metadata(initial_backend_key)
            effective_red = self.redirector_pool.active_redirector
            effective_backend_key = self.build_url(key, redirector=effective_red)
            self.redirector_url = effective_red
            return backend.__class__.__name__, effective_backend_key
        except (
            strax.DataNotAvailable,
            FileNotFoundError,
            OSError,
            IOError,
            TimeoutError,
            ConnectionError,
        ) as e:
            if allow_incomplete and not self.rucio_mode:
                temp_backend_key = f"{initial_backend_key}_temp"
                try:
                    backend.get_metadata(temp_backend_key)
                    effective_red = self.redirector_pool.active_redirector
                    effective_backend_key = f"{self.build_url(key, redirector=effective_red)}_temp"
                    self.redirector_url = effective_red
                    return backend.__class__.__name__, effective_backend_key
                except (
                    strax.DataNotAvailable,
                    FileNotFoundError,
                    OSError,
                    IOError,
                    TimeoutError,
                    ConnectionError,
                ):
                    pass
            raise strax.DataNotAvailable(
                f"Target data not available across any redirector for key {key}. "
                f"Last error: {e}"
            ) from e
