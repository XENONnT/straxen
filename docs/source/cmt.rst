Corrections Management Tool (CMT)
=================================
Corrections Management Tool (CMT) is a centralized tool that allows to store, query and retrieve information about detector effects (corrections) where later the information is used at the event building process to remove (correct) such effects for a given data type.
In specific CMT is a class within `strax <https://github.com/AxFoundation/strax/blob/master/strax/corrections.py>`_, the information is stored in MongoDB as document with a ``pandas.DataFrame()`` format and with a ``pandas.DatetimeIndex()`` this allows track time-dependent information as often detector conditions change over time. CMT also adds the functionality to differentiate between ONLINE and OFFLINE versioning, where ONLINE corrections are used during online processing and ,therefore, changes in the past are not allowed and OFFLINE version meant to be used for re-processing where changes in the past are allow.


CMT in straxen
--------------
A customized CMT can be implemented given the experiment software, in the case of straxen, experiment specifics can be added to CMT. To set CMT accordingly to straxen a class `CorrectionsManagementService() <https://github.com/XENONnT/straxen/blob/master/straxen/corrections_services.py>`_ allows the user to query and retrieve information. This class uses the start time of a given run to find the corresponding information and version for a given correction. For every correction user must set the proper configuration in order to retrieve the information, the syntax is the following ``my_configuration = (“my_correction”, “version”, True)`` the first part correspond to the string of the correction, then the version, it can be either an ONLINE version or OFFLINE version and finally the boolean correspond to the detector configuration (1T or nT). 
In the case of straxen there are several plug-ins that call CMT to retrieve information, in that case, the configuration option is set by the ``strax.option()`` and the information is retrieve in `set()` via the function `straxen.get_correction_from_cmt()` and example is shown below where the electron life time is retrieve for a particular run ID, using the ONLINE version for the detector configuration nT=True.


.. code-block:: python

    import straxen
    elife_conf = ("elife", "ONLINE", True)
    elife = straxen.get_correction_from_cmt(run_id, elife_conf)


An experiment specific option that provide the ability to do bookkeeping for the different versions is the introduction of the concept of global versions, global version means a unique set of corrections, e.g. ``global_v3={elife[v2], s2_map[v3], s1_map[v3], etc}``. This is specially useful for the creation of different context where the user has can set all the corresponding configuration using a global version via ``apply_cmt_context()``. However the user must be aware that only local version are allow for individual configurations from straxen prior to ``0.19.0`` the user had only the option to use global version. 
