from .config_tests import check_urls
from .url_config import URLConfig
import warnings

WARN = True


@URLConfig.preprocessor
def sort_url_kwargs(url: str):
    """
    Reorders queries for urlconfigs to avoid hashing issues
    """

    global WARN

    if isinstance(url, str) and URLConfig.SCHEME_SEP in url:
        if url != URLConfig.format_url_kwargs(url) and WARN:
            warnings.warn("From straxen version 2.1.0 onward, URLConfig parameters"
                          "will be sorted alphabetically before being passed to the plugins,"
                          " this will change the lineage hash for non-sorted URLs. To load"
                          " data processed with non-sorted URLs, you will need to use an"
                          " older version.")

            WARN = False
        return URLConfig.format_url_kwargs(url)
    return url


@URLConfig.preprocessor
def check_url_with_dispatcher(url: str):
    # Classes regex dispatcher to check url expressions
    if isinstance(url, str):
        check_urls(url)
