from .regex_dispatcher import RegexDispatcher

import straxen
import re
import warnings

check_urls = RegexDispatcher("check_urls")


class URLWarning(UserWarning):
    pass


@check_urls.register(r"(.*)(.*xedocs.*)")
def confirm_xedocs_schema_exists_check(url: str):
    import xedocs

    # Selects the schema name in the URL and confirms it exists in the xedocs db
    if not re.findall(r"(?<=\://)[^\://]*(?=\?)", url)[0] in xedocs.list_schemas():
        warnings.warn(
            f"You tried to use a schema that is not part of the xedocs database. url: {url}",
            URLWarning,
        )


#@check_urls.register(r".*(list-to-array|'objects-to-dict').*")
@check_urls.register(r"(.*)(.*xedocs.*)")
def data_as_list_check(url: str):
    if "list-to-array" or "objects-to-dict" in url:
        if not straxen.URLConfig.kwarg_from_url(url, "as_list"):
            warnings.warn(
                f"When using the list-to-array or objects-to-dict protocol, you must include an as_list=True in the URL arguments. url: {url}",
                URLWarning)
        if not ("sort" in url):
            warnings.warn(
                f"When using the list-to-array or objects-to-dict protocol, you must include a sort argument in the URL. url: {url}",
                URLWarning)


@check_urls.register(r"(.*)(.*xedocs.*)")
def are_resources_needed_check(url: str):
    parent_class = get_parent_class(url)

    if ("BaseMap" in parent_class) or ("BaseResourceReference" in parent_class):
        if not ("resource://" in url):
            warnings.warn(
                f"A URL which requeres the resource:// was given however resource:// was not found within the URL, not data will be loaded. url: {url}",
                URLWarning,
            )


@check_urls.register(r"(.*)(.*xedocs.*)")
def itp_map_check(url: str):
    parent_class = get_parent_class(url)

    if "BaseMap" in parent_class:
        if not ("itp_map://" in url):
            warnings.warn(
                f"Warning, you are requesting a map file with this URL however, the protocol itp_map:// was not requested as part of the URL. The itp_map:// protocol is requiered for map corrections. url: {url}",
                URLWarning,
            )


@check_urls.register(r".*posrec_models.*")
def tf_check(url: str):
    if not ("tf://" in url):
        warnings.warn(
            f"Warning, you are requesting a position reconstruction model with this URL however, the protocol tf:// was not requested as part of the URL. The tf:// protocol is requiered for position reconstruction corrections. url: {url}",
            URLWarning,
        )


@check_urls.register(r"(.*)(.*xedocs.*)")
def url_attr_check(url: str):
    if not ("attr" in url):
        warnings.warn(
            f"A URL without an 'attr' argument was given, as a result, instead of a value, list of values or other files, you will get a document, which cannot be used to process data. url: {url}",
            URLWarning,
        )


@check_urls.register(r"(.*)(.*xedocs.*)")
def url_version_check(url: str):
    if not ("version" in url):
        warnings.warn(
            f"A URL without a 'version' argument was given, as a result, to use a url protocol to get a correction a version of said correcection is requiered. url: {url}",
            URLWarning)


@check_urls.register(r".*fdc_maps.*")
def url_fdc_check(url: str):
    if not ("scale_coordinates" in url):
        warnings.warn(
            f"A URL for fdc was given without a [scale_coordinates] argument. This could lead to issues when reprocessing data. Please include the scaling in the URL. url: {url}",
            URLWarning,
        )


@check_urls.register(r".*objects-to-dict.*")
def tf_dict_attributes(url: str):
    if not ("key_attr" in url):
        warnings.warn(
            f"Warning, you are requesting a correction in the form of a dictionary with this URL however, you did not choose keys for the dictionary with [key_attr]. Please insert [key_attr=] in your url for the keys. url: {url}",
            URLWarning,
        )
    if not ("value_attr" in url):
        warnings.warn(
            f"Warning, you are requesting a correction in the form of a dictionary with this URL however, you did not include [value_attr] in your url which is needed for dictionry outputs. Please insert [key_attr=] in your url for the keys. url: {url}",
            URLWarning,
        )


def get_parent_class(url: str):
    """Finds a xedocs schema and returns the class used to make the schema."""
    import xedocs

    schema = re.findall(r"(?<=\://)[^\://]*(?=\?)", url)[0]
    xedocs_class = xedocs.find_schema(schema).__mro__
    parent_class = [parent.__name__ for parent in xedocs_class]

    return parent_class
