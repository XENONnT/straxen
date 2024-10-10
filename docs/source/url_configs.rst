
URLConfig options
=================
The URLConfig class was designed to make it easier have complex plugin configuration.
A plugin may require a rich object (such as a TensorFlow model), loading a file, or a run dependent value for its calculation.
While its perfectly reasonable to perform all of these operations in the plugins `setup()` method,
some operations such as loading files and looking up CMT values tend to repeat themselves in many plugins leading to code duplication.
Having the same code duplicated in many plugins can be very difficult to maintain or improve,
with the added annoyance that changing this behavior requires editing the plugin code.
The URLConfig provides a consistent way to define such behaviors at runtime via a URL string.
The URL is like a recipe for how the config value should be loaded when it is needed by the plugin.
Small snippets of code for loading a configuration can be registered as protocols and can be used by all plugins.
This allows you to keep the plugin code clean and focused on the processing itself,
without mixing in details of how to load the configuration data which tends to change more frequently.


The main goals of the URLConfig:

- More flexibility in switching between CMT, get_resource, and static configuration values.
- Remove logic of how to fetch and construct configuration objects from the plugin to improve purity (computational logic only) and maintainability of the plugins.
- Make unit testing easier by separating the logic that uses the configuration from the logic that fetches its current value.
- Increase the expressivity of the CMT values (descriptive string instead of opaque tuple)
- Remove need for hardcoding of special treatment for each correction in CMT when reading values.

A concrete plugin example
-------------------------

**The old way loading a TF model**

.. code-block:: python

    @export
    @strax.takes_config(
        strax.Option('min_reconstruction_area',
                    help='Skip reconstruction if area (PE) is less than this',
                    default=10),
        strax.Option('n_top_pmts', default=straxen.n_top_pmts,
                    help="Number of top PMTs")
    )
    class PeakPositionsBaseNT(strax.Plugin):

        def setup(self):
            self.model_file = self._get_model_file_name()
            if self.model_file is None:
                warn(f'No file provided for {self.algorithm}. Setting all values '
                    f'for {self.provides} to None.')
                # No further setup required
                return

            # Load the tensorflow model
            import tensorflow as tf
            if os.path.exists(self.model_file):
                print(f"Path is local. Loading {self.algorithm} TF model locally "
                    f"from disk.")
            else:
                downloader = straxen.MongoDownloader()
                try:
                    self.model_file = downloader.download_single(self.model_file)
                except utilix.mongo_storage.CouldNotLoadError as e:
                    raise RuntimeError(f'Model files {self.model_file} is not found') from e
            with tempfile.TemporaryDirectory() as tmpdirname:
                tar = tarfile.open(self.model_file, mode="r:gz")
                tar.extractall(path=tmpdirname)
                self.model = tf.keras.models.load_model(tmpdirname)

        def _get_model_file_name(self):
            config_file = f'{self.algorithm}_model'
            model_from_config = self.config.get(config_file, 'No file')
            if model_from_config == 'No file':
                raise ValueError(f'{__class__.__name__} should have {config_file} '
                                f'provided as an option.')
            if isinstance(model_from_config, str) and os.path.exists(model_from_config):
                # Allow direct path specification
                return model_from_config
            if model_from_config is None:
                # Allow None to be specified (disables processing for given posrec)
                return model_from_config

            # Use CMT
            model_file = straxen.get_correction_from_cmt(self.run_id, model_from_config)
            return model_file


Notice how all the details on how to fetch the model file and convert it to a python object that is actually needed, is all hardcoded into the plugin. This is not desirable, the plugin should contain processing logic only.

**How this could be refactored using `strax.URLConfig`:**

.. code-block:: python

    class PeakPositionsMLP(PeakPositionsBaseNT):
        tf_model_mlp = straxen.URLConfig(
            default=f'tf://'
                    f'resource://'
                    f'cmt://{algorithm}_model'
                    f'?version=ONLINE'
                    f'&run_id=plugin.run_id'
                    f'&fmt=abs_path',
        help='MLP model. Should be opened using the "tf" descriptor. '
             'Set to "None" to skip computation',
        cache=3,
    )

The details of where the model object is taken from can be determined by setting the model key of the context config
The URL is the object being hashed, so it is important to only use pure URLs i.e the same URL should always refer to the same resource.

The URL is evaluated recursively in the following order:
  1) **?version=ONLINE&run_id=plugin.run_id&fmt=abs_path** - Query is parsed and substituted (plugin.* are replaced with plugin attributes as evaluated at runtime) the values are then passed as keyword arguments to any protocols that include them in their signature. Everythin after the rightmost `?` character is considered the keyword arguments for the protocols.
  2) **cmt://** - Loads value from CMT, in this case it loads the name of the resource encoding the keras model.
  3) **resource://** - Loads a xenon resource by name (can also load web URLs), in this case returns a path to the file.
  4) **tf://** - Loads a TF model from a path

**Important** The URL arguments are sorted before they are passed to the plugin so that hashing is not sensitive to the order of the arguments.
This is important to remember when performing tests.
All of the actual code snippets for these protocols are shared among all plugins.

Adding new protocols
--------------------

As an example lets look at some actual protocols in `url_config.py`


.. code-block:: python

    @URLConfig.register('format')
    def format_arg(arg: str, **kwargs):
        """apply pythons builtin format function to a string"""
        return arg.format(**kwargs)


    @URLConfig.register('itp_map')
    def load_map(some_map, method='WeightedNearestNeighbors', **kwargs):
        """Make an InterpolatingMap"""
        return straxen.InterpolatingMap(some_map, method=method, **kwargs)


    @URLConfig.register('bodega')
    def load_value(name: str, bodega_version=None):
        """Load a number from BODEGA file"""
        if bodega_version is None:
            raise ValueError('Provide version see e.g. tests/test_url_config.py')
        nt_numbers = straxen.get_resource("XENONnT_numbers.json", fmt="json")
        return nt_numbers[name][bodega_version]["value"]


    @URLConfig.register('tf')
    def open_neural_net(model_path: str, **kwargs):
        # Nested import to reduce loading time of import straxen and it not
        # base requirement
        import tensorflow as tf
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'No file at {model_path}')
        with tempfile.TemporaryDirectory() as tmpdirname:
            tar = tarfile.open(model_path, mode="r:gz")
            tar.extractall(path=tmpdirname)
            return tf.keras.models.load_model(tmpdirname)

As you can see its very easy to define new protocols, once its defined you can use it in any URL!


Config preprocessors
--------------------

In some cases it makes sense to run some code and maybe modify a config value during
the plugin configuration initialization. This will result in the config being replaced
completely in the ``plugin.config`` dictionary and the modified value being hashed instead of the original value.
The preprocessor function you register can accept any or all of the following keyword arguments: name, run_id, run_defaults, set_defaults.
These keyword arguments will be passed their values when the function is invoked.

A simple example would be if you want to support string formatting in configs:

.. code-block:: python

    @straxen.URLConfig.preprocessor
    def formatter(config, **kwargs):
        if not isinstance(config, str):
            return config
        try:
            config = config.format(**kwargs)
        except:
            pass
        return config

This preprocessor will run on all configs and if any of them are strings it will
attempt to run the builtin ``format`` function on them with all the keyword arguments available at that time.

You can also control the order in which preprocessors are run in cases where multiple
functions are registered by passing the ``precedence=N`` argument to the decroator, where N is the priority.
Higher precedence functions are run first.

**WARNINGS**:

* Using the run_id to set the value of the config will result in a different lineage_hash for each run. This may be useful in some cases but can be very difficult to keep track of with data managment tools.
* Preprocessor functions will run on **all** configs. If you want to only affect a specific config, make sure your function accepts the ``name`` keyword argument and that the function checks the name matches before it runs its logic.

Helper functions
----------------

There are a number of helper functions to help you writing URLs:

* ``straxen.URLConfig.print_protocols()`` - prints out the registered protocols, their description and call signature.
* ``straxen.URLConfig.print_preprocessors()`` -  prints out the registered preprocessor precedence, their description and call signature.
* ``straxen.URLConfig.print_status()`` - Calls both ``print_protocols()`` and ``print_preprocessors()``
