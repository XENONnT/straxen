import strax
from unittest import TestCase


class PluginTestAccumulator:
    """
    Accumulator for test functions for unit-testing such that all plugin
    related unit tests can be run on the same data within a single
    unit-test.

    Use example:
    ```python
        from _core import PluginTestAccumulator


        @PluginTestAccumulator.register('test_example')
        def test_example(self, # You should always accept self as an argument!
                        ):
            raise ValueError('Test failed')
    ```
    """

    # See URLConfigs for the original inspiration.
    @classmethod
    def register(cls, test_name, func=None):
        def wrapper(func):
            if not isinstance(test_name, str):
                raise ValueError('test_name name must be a string.')
            if not test_name.startswith('test'):
                raise ValueError(f'Tests should start with test_.., '
                                 f'got {test_name} for {func}')
            if hasattr(cls, test_name):
                raise ValueError(f'{test_name} already used!')
            setattr(cls, test_name, func)
            return func

        return wrapper(func) if func is not None else wrapper


class PluginTestCase(TestCase):
    """Class for type hinting of PluginTest"""
    run_id: str
    st: strax.Context
