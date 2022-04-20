# Test for plugins

In this directory, we accumulate tests for the plugins we are using. 
This makes data-manipulation and plugin configuration tests development easy.

## Writing new tests
Write a new test by adding a `.py` file and write a function that should be run
in the testing suite by adding this decorator:
```python
from _core import PluginTestAccumulator


@PluginTestAccumulator.register('test_example')
def test_example(self, # You should always accept self as an argument!
                ):
    raise ValueError('Test failed')
```

Make sure that the new `.py`-file is imported in the `test_plugins.py` file 
(see explanation below).

## Example
See for example this ![event_building.py](event_building.py)-file. Here we add 
a test that works on event_basics (by changing a few configurations and 
checking that the output is roughly what we expect.


## Why this organization (technical - only for developers)
At the time of writing, there are ~132 different dependencies. Testing them 
does require quite some computational power if we all want to compute them, 
especially if we want to start computation from scratch (to avoid trying to a 
test that turns out to rely on data that was scrambled).

A few design consideration were taken into account:
 - We want a single class with a `tearDownClass`-method to clean up the data 
   after running. If we would have many classes doing this, you would end up 
   producing the same data many times. If you don't have a `tearDownClass` with
   storage cleanup, you could end up working on cached data which can cause a 
   lot of frustration  https://github.com/XENONnT/straxen/pull/923.
 - Add a modular approach where new tests can be added (in a for loop). 
   While [subtests](https://docs.python.org/3/library/unittest.html#distinguishing-test-iterations-using-subtests) 
 - should be a nice tool for doing this, it turns out to be not as transparent 
   as it claims. Furthermore, you somehow still need to have the entire class 
   in one file, whereas we'd lile to split these tests into small, targeted 
   files.
 - In order to accumulate the test functions, we made a dedicated class 
   `PluginTestAccumulator` that just is a dump for the tests that we write in
   various places. We can't very cleanly re-import `unittest.TestCase`-classes,
   it seems to break the `pytest` (not doing proper setup- and teardowns).

With this setup, we can very easily write tests at any level of data without 
doing a lot of re-computes of already stored data while providing a simple 
infrastructure for developers.
