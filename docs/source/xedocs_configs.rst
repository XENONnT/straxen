Xedocs context configs
======================

The xedocs package manages a versioned collection of context configs. 
This is useful for applying a predefined set of configs to the context in a reproducible way.
The straxen context builder function `straxen.contexts.xenonnt` accepts a `xedocs_version` argument, 
if passed, straxen will attempt to load all configs from the given version from the xedocs context_configs database.
