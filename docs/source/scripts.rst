Straxen scripts
===================
Straxen comes with
`several scripts <https://github.com/XENONnT/straxen/tree/master/bin>`_
that allow common uses of straxen. Some of these scripts are designed
to run on the DAQ whereas others are for common use cases. Each of the
scripts will be briefly discussed below:

straxer
-------
``straxer`` is the most useful straxen script for regular users. Allows data to be
generated in a script format. Especially useful for reprocessing data
in batch jobs.

For example a user can reprocess the data of run ``012100`` using the
following command up to ``event_info_double``.

.. code-block:: bash

    straxer 012100 --target event_info_double

For more information on the options, please refer to the help:

.. code-block:: bash

    straxer --help


ajax [DAQ-only]
----------------
The DAQ-cleaning script. Data is stored on the DAQ such that other tools
like `admix <https://github.com/XENONnT/admix>`_ may ship the data to
distributed storage. A portion of the high level data is stored on the DAQ
for diagnostic purposes for longer periods of time. ``ajax`` removes this
data if needed.
The ``ajax`` script looks for data on the eventbuilders
that can be deleted because at least one of the following reasons:

 - A run has been "abandoned", this means that there is no further use
   for this data, e.g. a board failed during a run, there is no point in
   keeping a run where part of the data on the DAQ.
 - The live-data (intermediate DAQ format, even more raw than raw-records) has
   been successfully processed. Therefore remove this intermediate datakind from
   daq.
 - A run has been abandoned but there is live-data still on the DAQ-bugger.
 - Data is "unregistered" (not in the runsdatabase),
   this only occurs if DAQ-experts perform tests on the DAQ.
 - Since bootstrax runs on multiple hosts, some of the data may appear to be
   stored more than once since a given bootstrax instance could crash during it's processing.
   The data of unsucessful processings should be removed by ``ajax``.
 - Finally ``ajax`` also checks if all the entries that are in the database are also on the host still
   This sanity check catches any potential issues in the data handling by admix.


bootstrax [DAQ-only]
--------------------
As the main DAQ processing script. This is discussed separately. It is only used for XENONnT.


fake_daq
------------------
Script that allows mimiming DAQ-processing by opening raw-records data.


microstrax
------------------
Mini strax interface that allows strax-data to be retrieved using HTTP requests
on a given port. This is at the time of writing used on the DAQ as a pulse viewer.


refresh_raw_records
-------------------
Updates raw-records from old strax versions. This data is of a different
format and needs to be refreshed before it can be opened with more recent
versions of strax.

*Last updated 2021-05-07. Joran Angevaare*
