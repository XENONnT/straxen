Xedocs context configs
======================

The xedocs package manages a versioned collection of context configs. 
This is useful for applying a predefined set of configs to the context in a reproducible way.
Straxen registers a context method `context.apply_xedocs_configs` which can be used to load and apply 
the configs from a given version. Example:

.. code-block:: python
    
        import straxen
        st = straxen.contexts.xenonnt()
        st.apply_xedocs_configs('v9')


The straxen context builder function `straxen.contexts.xenonnt` also accepts a `xedocs_version` argument, 
if passed, straxen will attempt to load all configs from the given version from the xedocs context_configs database.
Example:

.. code-block:: python
    
        import straxen
        st = straxen.contexts.xenonnt(xedocs_version='v9')
