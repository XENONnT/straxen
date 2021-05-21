Setting up straxen
===================

To install straxen locally, follow these steps in a python 3.6, 3.7 or 3.8 environment:

1. `git clone https://github.com/XENONnT/straxen`
2. **Optional**. If you are NOT on the UChicago Midway analysis center, copy the file `/project2/lgrandi/xenonnt/.xenon_config` to your `$HOME`-directory. Even with this step, you will have very limited access to XENON data.
3. `pip install -e ./straxen`

The basic analysis tutorial will work outside midway even without step 2 -- we ship a tiny bit of test data with straxen for this purpose. Not much else will work, in particular, anything that requires the XENON runs database will fail.

To update straxen to a new version, execute `git pull` in the straxen directory. You may also have to update strax itself occasionally with `pip install strax --upgrade``.

Frozen installation
--------------------
Instead of step 3, you may want to do try a 'frozen install' with `pip install ./straxen` (without the `-e`). This means the straxen code gets copied to a different place. Edits to the source files will only take effect after you run `pip install straxen` again.

Currently you are likely to encounter problems loading the test data in this mode --  I've been too lazy to update the manifest accordingly.

Moreover, you must be extra careful if you are outside midway: step 2 really has to be done *before* step 3, not afterwards.
