class PluginTestAccumulator:
    # See URLConfigs for the original inspiration.
    @classmethod
    def register(cls, test_name, func=None):
        def wrapper(func):
            if not isinstance(test_name, str):
                raise ValueError('test_name name must be a string.')
            if not test_name.startswith('test'):
                raise ValueError(f'Tests should start with test_.., '
                                 f'got {test_name} for {func}')

            setattr(cls, test_name, func)
            return func

        return wrapper(func) if func is not None else wrapper
