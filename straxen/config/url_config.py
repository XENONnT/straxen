# Moved URL configs to strax.
# Keep the old import paths working for backwards compatibility.
# E.g., for straxen.config.url_config.URLConfig, in straxen, cutax and xedocs.
from strax.url_config import (  # noqa: F401
    _CACHES,
    URLConfig,
    clear_config_caches,
    config_cache_size_mb,
    parse_val,
)

__all__ = ["URLConfig", "clear_config_caches", "config_cache_size_mb"]
