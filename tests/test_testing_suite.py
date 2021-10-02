from straxen import _is_on_pytest


def test_testing_suite():
    """
    Make sure that we are always on a pytest when this is called. E.g.
        pytest tests/test_testing_suite.py

    A counter example would e.g. be:
        python tests/test_testing_suite.py
    """
    assert _is_on_pytest()


if __name__ == '__main__':
    test_testing_suite()
