Extending Straxen using Entry Points
====================================
Straxen has multiple methods for runtime extension of it functionality
such as adding context methods, new plugins, URLConfig protocols and mini-analyes.
These extensions can be registered at runtime by the user when needed.
Sometimes its useful to place extra features/functionality in a separate package and have the functionality
registered automatically when straxen is imported. For this python has a mechanism called entry points.
Entry points are a way to extend the functionality of a package without having to modify the package itself.
This is done by adding a section to the package's setup.py file. This section is called entry_points and
is a dictionary where the keys are the entry point group and the values are a list of strings with the
entry point name and the object reference. For example in you setup.py:


.. code-block::

    entry_points={
        'straxen': [
            '_ = my_package.my_module:register_my_plugin',
        ],
    }


or if you have a pyproject.toml file:

.. code-block::

    [tool.poetry.plugins."straxen"]
    "_" = "my_package.my_module:register_my_plugin"


The right hand side of the entry point is a reference to the object that should be registered.
It can be a module or a callable. If it is a callable,
straxen will call it on import, otherwise the module will simply be imported.
