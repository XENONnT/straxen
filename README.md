# straxen
Streaming analysis for XENON(nT)


[![Join the chat at https://gitter.im/AxFoundation/strax](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/AxFoundation/strax?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Straxen is the analysis framework for XENONnT, built on top of the generic [strax framework](https://github.com/AxFoundation/strax). Currently it is configured for analyzing XENON1T data.

For installation instructions and usage information, please see  the [straxen documentation](https://straxen.readthedocs.io).

### Installation

  1. `git clone https://github.com/XENONnT/straxen`
  2. **Optional**. If you want access to the XENON data but you are NOT on the UChicago Midway analysis center:
     * Copy the file `/home/aalbers/xenon_secrets.py` from midway to  `./straxen/straxen/`, i.e. to the same directory as `xenon_context.py`.
  3. `pip install -e straxen`

Instead of step 3, you can try a 'frozen install' with `pip install straxen`, but you are likely to encounter problems loading the test data (I've been too lazy to update the manifest). Moreover, in a frozen install outside midway, you must do step 2 before step 3, not afterwards. 

The demo notebook will run without the XENON secrets (step 2), though almost nothing else will. 


For more information, please see the [strax documentation](https://strax.readthedocs.io).
