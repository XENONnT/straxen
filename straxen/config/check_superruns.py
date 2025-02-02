import strax
import straxen


def get_components_wrapper(func):
    """Check whether all configs for superruns-allowed plugins are the same for subruns."""

    def wrapper(self, run_id, targets=tuple(), **kwargs):
        is_superrun = run_id.startswith("_")
        if (
            not is_superrun
            or not self.context_config["check_superrun_configs"]
            or kwargs.get("combining", False)
        ):
            return func(self, run_id=run_id, targets=targets, **kwargs)

        configs = self.time_dependent_configs(run_id, targets)
        if not configs:
            return func(self, run_id=run_id, targets=targets, **kwargs)
        superrun_hashes = {run_id: self.hashed_url_configs(configs)}

        # collect all url config of all subruns and convert them into hash
        sub_run_spec = self.run_metadata(run_id, projection="sub_run_spec")["sub_run_spec"]
        for sub_run_id in sub_run_spec:
            configs = self.time_dependent_configs(sub_run_id, targets)
            superrun_hashes[sub_run_id] = self.hashed_url_configs(configs)

        # check whether all subruns have the same configs
        seen = strax.deterministic_hash(superrun_hashes[run_id])
        for sub_run_id, value in superrun_hashes.items():
            if seen != strax.deterministic_hash(value):
                diff = [k for k in value if value[k] != superrun_hashes[run_id][k]]
                raise ValueError(
                    "Superrun configs are not the same for all subruns. "
                    f"Specifically, the following configs are different: {diff}."
                )

        return func(self, run_id=run_id, targets=targets, **kwargs)

    return wrapper


# Manually decorate context class
_strax_context_decorated = True
if "check_superrun_configs" not in strax.Context.takes_config:
    strax.Context = strax.takes_config(
        strax.Option(
            name="check_superrun_configs",
            default=True,
            type=bool,
            help="If True, check whether all subruns' config are the same.",
        ),
    )(strax.Context)
    _strax_context_decorated = False
if "plugin_attr_convert" not in strax.Context.takes_config:
    strax.Context = strax.takes_config(
        strax.Option(
            name="plugin_attr_convert",
            default=("run_id", "algorithm"),
            type=(list, tuple),
            help="The attributes that should be get from the plugin.",
        ),
    )(strax.Context)
    _strax_context_decorated = False
if not _strax_context_decorated:
    # Overwrite get_components method
    strax.Context.get_components = get_components_wrapper(strax.Context.get_components)


@strax.Context.add_method
def time_dependent_configs(self, run_id, targets, superrun_only=True):
    targets = strax.to_str_tuple(targets)
    plugins = self._get_plugins(targets=targets, run_id=run_id)
    configs = dict()
    for data_type, plugin in plugins.items():
        if not superrun_only or (plugin.allow_superrun and superrun_only):
            for k, v in plugin.config.items():
                if not isinstance(v, str):
                    continue
                configs[k] = (v, plugins[data_type])
    return configs


@strax.Context.add_method
def hashed_url_configs(self, configs):
    """Convert all url into string and hash them."""
    # get all configs for superruns-allowed plugins if they are str
    hash = dict()
    for key, value in configs.items():
        if key == "superrun_test_config_a":
            print("HERE")
        url, plugin = value
        url = straxen.URLConfig.format_url_kwargs(url)
        arg, extra_kwargs = straxen.URLConfig.split_url_kwargs(url)
        # only extract value, avoid attr=map
        if "attr" in extra_kwargs:
            extra_kwargs["attr"] = "value"
        # transform run_id and algorithm from plugin
        # run_id and algorithm are the only two keys that
        # can be extracted from plugin for now
        # if later we have more keys need to be converted from plugin
        # to make xedocs work, we need to add them here
        for k in self.context_config["plugin_attr_convert"]:
            if k in extra_kwargs:
                extra_kwargs[k] = getattr(
                    plugin,
                    extra_kwargs[k].partition(straxen.URLConfig.NAMESPACE_SEP)[-1],
                )
        # filter out plugin related kwargs
        _extra_kwargs = dict()
        for k, v in extra_kwargs.items():
            flag = not isinstance(v, str)
            flag |= isinstance(v, str) and straxen.URLConfig.PLUGIN_ATTR_PREFIX not in v
            if flag:
                _extra_kwargs[k] = v
        url = straxen.URLConfig.format_url_kwargs(arg, **_extra_kwargs)
        # replace all possible missing attr like in runstart://plugin.run_id
        # the remaining plugin related attr have not been replaced because they are not kwargs
        for k in self.context_config["plugin_attr_convert"]:
            if f"plugin.{k}" in url:
                url = url.replace(f"plugin.{k}", getattr(plugin, k))
        if "resource" in url:
            protocol, arg, kwargs = straxen.URLConfig.url_to_ast(url)
            while protocol != "resource":
                protocol, arg, kwargs = arg
            if isinstance(arg, tuple):
                url = straxen.URLConfig.ast_to_url(arg)
            else:
                url = arg
        evaluated = straxen.URLConfig.evaluate_dry(url)
        try:
            strax.hashablize(evaluated)
        except TypeError as e:
            raise ValueError(
                f"Cannot hash url {url} converted from config {key} in superrun safeguard. "
                f"Because its evaluated value not hashable: {evaluated}. "
                f"Error: {e}"
            )
        hash[key] = strax.deterministic_hash(evaluated)
    return hash
