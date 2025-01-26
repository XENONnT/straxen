import strax


def get_components_wrapper(func):
    """Check whether all configs for superruns-allowed plugins are the same for subruns."""

    def wrapper(self, run_id, targets=tuple(), **kwargs):
        is_superrun = run_id.startswith("_")
        if not is_superrun:
            return func(self, run_id=run_id, targets=targets, **kwargs)

        plugins = self._get_plugins(
            targets=targets, run_id=run_id, chunk_number=kwargs.get("chunk_number", None)
        )
        configs = dict()
        for v in plugins.values():
            if v.allow_superrun:
                configs.update(v.config)
        configs = dict((k, v) for k, v in configs.items() if isinstance(v, str))
        # TODO:
        # stop at resource
        # remove attr=map
        # remove plugin
        if not configs:
            return func(self, run_id=run_id, targets=targets, **kwargs)

        # sub_run_spec = self.run_metadata(run_id, projection="sub_run_spec")["sub_run_spec"]

    return wrapper


strax.Context.get_components = get_components_wrapper(strax.Context.get_components)
