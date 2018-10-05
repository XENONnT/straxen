# straxen
Streaming analysis for XENON


[![Join the chat at https://gitter.im/AxFoundation/strax](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/AxFoundation/strax?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This repository hosts the XENON-specific code for the data analysis framework strax. 


### Installation

  1. `git clone https://github.com/XENONnT/straxen`
  2. Optional: for access to the XENON runs data:
     * Copy the file `/home/aalbers/xenon_secrets.py` from midway to  `./straxen/straxen/`, i.e. to the same directory as `xenon_context.py`.
  3. `pip install -e straxen` (developer mode) or `pip install straxen` (frozen install)

If you choose a frozen install, and you want access to the XENON data, you must do step 2 **before** step 3. (That's why it's labeled step 2.) 

The demo notebook will run without the XENON secrets (step 2), though almost nothing else will. 


For more information, please see the [strax documentation](https://strax.readthedocs.io).
