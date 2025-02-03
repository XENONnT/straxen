from .regex_dispatcher import RegexDispatcher

import straxen
import re
import warnings

check_urls = RegexDispatcher("check_urls")


class URLWarning(UserWarning):
    pass


@check_urls.register(r"(.*)(.*cmt.*)")
def not_cmt_check(url: str):
    raise NotImplementedError("Error, the cmt protocol is removed. Please use xedocs instead.")


@check_urls.register(r"(.*)(.*xedocs.*)")
def confirm_xedocs_schema_exists_check(url: str):
    import xedocs

    # Selects the schema name in the URL and confirms it exists in the xedocs db
    if not re.findall(r"(?<=\://)[^\://]*(?=\?)", url)[0] in xedocs.list_schemas():
        warnings.warn(
            f"You tried to use a schema that is not part of the xedocs database. url: {url}",
            URLWarning,
        )


# @check_urls.register(r".*(list-to-array|'objects-to-dict').*")
@check_urls.register(r"(.*)(.*xedocs.*)")
def data_as_list_check(url: str):
    if ("list-to-array" in url) or ("objects-to-dict" in url):
        if not straxen.URLConfig.kwarg_from_url(url, "as_list"):
            warnings.warn(
                "When using the list-to-array or objects-to-dict protocol, "
                f"you must include an as_list=True in the URL arguments. url: {url}",
                URLWarning,
            )
        if not ("sort" in url):
            warnings.warn(
                "When using the list-to-array or objects-to-dict protocol, "
                f"you must include a sort argument in the URL. url: {url}",
                URLWarning,
            )


@check_urls.register(r"(.*)(.*xedocs.*)")
def are_resources_needed_check(url: str):
    parent_class = get_parent_class(url)

    if ("BaseMap" in parent_class) or ("BaseResourceReference" in parent_class):
        if not ("resource://" in url) and straxen.URLConfig.kwarg_from_url(url, "attr") != "map":
            warnings.warn(
                "A URL which requests the resource:// was given. However resource:// "
                f"was not found within the URL, not data will be loaded. url: {url}",
                URLWarning,
            )


@check_urls.register(r"(.*)(.*xedocs.*)")
def itp_map_check(url: str):
    parent_class = get_parent_class(url)

    if "BaseMap" in parent_class:
        if (
            not ("itp_map://" in url)
            and straxen.URLConfig.kwarg_from_url(url, "attr") != "map"
            and not ("tf://" in url)
        ):
            warnings.warn(
                "Warning, you are requesting a map file with this URL. However, the protocol "
                "itp_map:// was not requested as part of the URL. "
                f"The itp_map:// protocol is requiered for map corrections. url: {url}",
                URLWarning,
            )


@check_urls.register(r".*posrec_models.*")
def posrec_models_check(url: str):
    if not ("tf://" in url) and not ("jax://" in url):
        warnings.warn(
            "Warning, you are requesting a position reconstruction model with this URL. However, "
            "the protocol tf:// and jax:// were not requested as part of the URL. "
            "The tf:// or jax://protocol is requiered for position reconstruction corrections. "
            f"url: {url}",
            URLWarning,
        )


@check_urls.register(r".*tf.*")
def keras_check(url: str):
    if not ("readable=True" in url):
        raise ValueError(
            "Error, you are requesting a keras model with this URL. However, "
            "the protocol readable=True was not requested as part of the URL. "
            "The readable=True protocol is requiered for keras models. "
            "Because keras load_model only accepts .keras files. "
            f"url: {url}"
        )


@check_urls.register(r"(.*)(.*xedocs.*)")
def url_attr_check(url: str):
    if not ("attr" in url):
        warnings.warn(
            "A URL without an 'attr' argument was given, as a result, instead of a value, "
            "list of values or other files, you will get a document, "
            f"which cannot be used to process data. url: {url}",
            URLWarning,
        )


@check_urls.register(r"(.*)(.*xedocs.*)")
def url_version_check(url: str):
    if not ("version" in url):
        warnings.warn(
            "A URL without a 'version' argument was given, as a result, to use a url protocol "
            f"to get a correction a version of said correcection is requiered. url: {url}",
            URLWarning,
        )


@check_urls.register(r".*objects-to-dict.*")
def dict_attributes(url: str):
    if not ("key_attr" in url):
        warnings.warn(
            "Warning, you are requesting a correction in the form of a dictionary with "
            "this URL. However, you did not choose keys for the dictionary with [key_attr]. "
            f"Please insert [key_attr=] in your url for the keys. url: {url}",
            URLWarning,
        )
    if not ("value_attr" in url):
        warnings.warn(
            "Warning, you are requesting a correction in the form of a dictionary with "
            "this URL. However, you did not include [value_attr] in your url which is needed "
            "for dictionry outputs. Please insert [key_attr=] "
            f"in your url for the keys. url: {url}",
            URLWarning,
        )


def get_parent_class(url: str):
    """Finds a xedocs schema and returns the class used to make the schema."""
    import xedocs

    schema = re.findall(r"(?<=\://)[^\://]*(?=\?)", url)[0]
    xedocs_class = xedocs.find_schema(schema).__mro__
    parent_class = [parent.__name__ for parent in xedocs_class]

    return parent_class
