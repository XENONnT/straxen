# XENONnT plugins

This subfolder holds all the plugins for the XENONnT experiment. The plugins are structured using the following rules:

- Each datakind gets its own subfolder
- Each plugin gets its own module
- Imports of plugins are ordered by stream (TPC/Neutron Veto/Muon Veto/High Energy)
- Shared code of several plugins is either:
    - Shared by re-importing
    - Shared via an underscored (private) module (see `straxen/straxen/plugins/peaks/_peak_positions_base.py`)
- In case of a multi-output plugin, there may be several datakinds that are produced.
  In this case, the plugin is stored in a folder of the _first_ datakind (see e.g. raw_records, peaklets)
- Defaults that are shared are stored in `defaults.py` and capitalized
