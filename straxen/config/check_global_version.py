import strax
import straxen


def check_global_version_wrapper(func):
    """Check whether all configs for superruns-allowed plugins are the same for subruns."""

    def wrapper(self, run_id, targets=tuple(), **kwargs):
        if (
            not self.context_config["check_global_version_configs"]
            or "xedocs_version" not in self.context_config
            or self.context_config["xedocs_version"] is None
            or self.context_config["xedocs_version"] == "global_ONLINE"
        ):
            return func(self, run_id=run_id, targets=targets, **kwargs)

        configs = self.versioned_configs(run_id, targets)
        if not configs:
            return func(self, run_id=run_id, targets=targets, **kwargs)

        for key, value in configs.items():
            url, plugin = value
            arg, extra_kwargs = straxen.URLConfig.split_url_kwargs(url)
            if extra_kwargs["version"] == "ONLINE":
                raise ValueError(
                    f"The global version is set to be {self.context_config['xedocs_version']}. "
                    f"But {plugin.__class__.__name__} is still using ONLINE version "
                    f"config {key}, which is {url}."
                )

        return func(self, run_id=run_id, targets=targets, **kwargs)

    return wrapper


# Manually decorate context class
_strax_context_check_global_version_decorated = True
if "check_global_version_configs" not in strax.Context.takes_config:
    strax.Context = strax.takes_config(
        strax.Option(
            name="check_global_version_configs",
            default=True,
            type=bool,
            help=(
                "If True, check whether version=ONLINE is not used "
                "when global version is not global_ONLINE."
            ),
        ),
    )(strax.Context)
    _strax_context_check_global_version_decorated = False
if not _strax_context_check_global_version_decorated:
    # Overwrite get_components method
    strax.Context.get_components = check_global_version_wrapper(strax.Context.get_components)


@strax.Context.add_method
def versioned_configs(self, run_id, targets, superrun_only=True):
    targets = strax.to_str_tuple(targets)
    plugins = self._get_plugins(targets=targets, run_id=run_id)
    configs = dict()
    for data_type, plugin in plugins.items():
        for k, v in plugin.config.items():
            if not isinstance(v, str):
                continue
            arg, extra_kwargs = straxen.URLConfig.split_url_kwargs(v)
            if "version" in extra_kwargs:
                configs[k] = (v, plugins[data_type])
    return configs
