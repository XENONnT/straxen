from .config_tests import check_urls
from .url_config import URLConfig


@URLConfig.preprocessor
def sort_url_kwargs(url: str):
    """Reorders queries for urlconfigs to avoid hashing issues."""

    if isinstance(url, str) and URLConfig.SCHEME_SEP in url:
        return URLConfig.format_url_kwargs(url)
    return url


@URLConfig.preprocessor
def check_url_with_dispatcher(url: str):
    # Classes regex dispatcher to check url expressions
    if isinstance(url, str):
        check_urls(url)
