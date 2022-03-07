##############################
Corrections Managment Tool 2.0
##############################

CMT2.0 is reimplementation from scratch of CMT.

The framework was development to solve a number of issues that came up using the CMT tool.


    - Rigid indexing - all corrections are indexed by time only, requiring individual corrections for each PMT in the case of the pmt gains. This also limits the corrections database to only store time-dependent values, requiring other solutions for storing corrections without a time dependence.
    - Shared indexing - all versions of a single correction share the same time index, requiring special values (null) as a hack to have independent intervals of validity for each version. This complicates updating interpolated values like the pmt gains.
    - Inefficient use of MongoDB - the database is used as an object store for storing pandas dataframes, requiring the upload of the entire dataframe (for all times/versions) when updating a correction.
    - Hardcoded corrections - adding a new correction requires adding a new global version and releasing a new straxen version.
    - Single value storage - Each correction can only hold a single value, this prevent adding extra data fields (eg errors) and metadata fields (eg analyst name, creating date, description etc ) requiring a separate git repo just to store the metadata.
    - Global versions as strings - global versions are stored as json strings in a dedicated collection, no time dependence and adding new corrections requires a new global version. global versions have special status in CMT requiring extra code just to manage this collection. 

Remote Dataframes
-----------------

At its core CMT was supposed to be a simple way to have shared dataframe-like collections of data
 that can be accessed by all analysts and enforces certain rules on writing new data. To this end,
 the RemoteDateframe was designed to behave similarly to a MultiIndex dataframe but selection operations
 are converted to database queries. The framework was designed to be extensible and allows adding support
 for special indexes or other data backends.

Currently supported data-backends:

    - MongoDB
    - Pandas dataframe
    - Http client

Currently implemented indexes:

    - Simple Index of type Integer/String/Float/Datetime, matches on exact values.
    - Interval Index Datetime/Integer, matches on interval overlap.
    - Sampled Index Datetime/Integer/Float, interpolation between sampled points, optional extrapolation.

To define access a remote dataframe you need to first define its schema. Lets say we want to store a collection
of versioned documents indexed by the experiment, detector and version

.. code-block:: python

    class ExampleSchema(rframe.BaseSchema):
        name = 'simple_dataframe'

        # The simplest index, matches on exact value. 
        # This is how we define a versioned document without 
        # time dependence
        experiment: Literal['1t', 'nt'] = rframe.Index()
        detector: Literal['tpc', 'nveto','muveto'] = rframe.Index()
        version: str = rframe.Index()

        # we can define as many fields as we like, each with its own type
        # defined using standard python annotations
        value: float
        error: float
        unit: str
        creator: str

    def pre_insert(self, db):
            # the base class implementation checks if any documents already exist at the index 
        # and raises an error if it does
            super().pre_insert(db)
            # here you can add any special logic to perform prior to inserts

    def pre_update(self, db, new):
            # the base class implementation raises an error if new values dont match
            # the current ones for this index. we allow replacing the current values
            # with identical ones because the current values may be inferred (i.e interpolated)
            # in which case we allow new documents with the interpolated values since that wont
            # change any interpolated values.
            super().pre_update(db, new)

            # add any extra logic/checks to perform here 


Once we have a schema, we can use it to build database queries on any of the supported data backends

.. code-block:: python

    import pymongo
    import pandas as pd

    db = pymongo.MongoClient()['cmt2']
    # or 
    db = pd.read_csv("pandas_dataframe.csv")

    doc = ExampleSchema.find(db, experiment=..., detector=..., version=...)


Alternatively we can use the ``RemoteDataframe`` class to access/store documents in any supported backend.

.. code-block:: python

    rf = rframe.RemoteFrame(ExampleSchema, db)

**Reading specific rows**

Rows can be accessed by calling the dataframe with the rows index values, using pandas-like indexing ``df.loc[idx]``, ``df.at[idx, column]``, ``df[column].loc[idx]`` or with the xarray style ``df.sel(index_name=idx)`` method

.. code-block:: python

    # These methods will al return an identical pandas dataframe

    df = rf.loc[experiment,detector, version]
    
    df = rf.sel(experiment=experiment, detector=detector, version=version)
    
    df = rf.loc[experiment,detector, version]
    
    # Access a specific column to get a series back
    df = rf['value'].loc[experiment,detector, version]
    df = rf.value.loc[experiment,detector, version]

    # pandas-style scalar lookup returns a scalar
    value = rdf.at[(experiment,detector, version), 'value']
    # or call the dataframe with the column as argyment and index values as keyword arguments
    value = rf('value', experiment=experiment, detector=detector, version=version)

**Slicing**

You can also omit indices to get results back matching all values of the omitted index

.. code-block:: python

    df = rf.sel(version=version)

    # or
    df = rf.loc[experiment, detector, :]

    # or
    df = rf.loc[experiment]

    # or pass a list a values you want to match on:
    df = rf.sel(version=[0,1], experiment=experiment)

    # Slicing is also supported
    df = rf.sel(version=slice(2,10), detector=detector)


The interval index also supports passing a tuple/slice/begin,end keywords to query all intervals overlapping the given interval

.. code-block:: python

    df = rf.sel(version=version, time=(time1,time2))
    df = rf.loc[version, time1:time2]
    df = rf.get(version=version, begin=time1, end=time2)


Corrections
-----------

Correction definitions should subclass the ``straxen.BaseCorrectionSchema`` or
 one of its subclasses and added via PR to straxen so that they can be used in processing.
 When subclassing a Correction class, you must give it a unique ``name`` attibute.

``BaseCorrectionSchema`` subclasses:

    - TimeSampledCorrection - indexed by version and time, where time is a datetime
    - TimeIntervalCorrection - indexed by version and time, where time is a interval of datetimes

Any subclass of ``BaseCorrectionSchema`` will automatically become available in the ``straxen.cframes`` namespace

.. code-block:: python

    rdfs = straxen.cframes.pmt_gains

    # specific remote dataframes can be accessed via dict-like access or attribute access by their name
    rf = straxen.cframes.pmt_gains
    # or
    rf = straxen.cframes['pmt_gains']

    df = rf.sel(version=..., detector=..., time=...)


Finding a correction document
-----------------------------

Corrections will query the mongodb correction database by default, if no explicit datasource is given.

.. code-block:: python
    
    drift_velocity = straxen.Bodega.find_one(field='drift_velocity', version='v1')
    
    # Returns a Bodega object with attributes value, description etc.
    drift_velocity.value

    all_v1_documents = straxen.Bodega.find(version='v1')

References
-----------

Some corrections are actually references, 
in this case there will be a .load() method to fetch the object being reference.

Examples:

.. code-block:: python

    # will return a reference to one or more correction documents
    ref = straxen.CorrectionReference.find_one(correction='pmt_gains', version=..., time=...)

    # will fetch the corrections being references
    pmt_gains = ref.load()

    # will return a reference to a resource (a FDC map)
    ref = straxen.FdcMapName.find_one(version=..., time=..., kind=...)

    # will return the map being referenced.
    fdc_map = ref.load()


The Corrections server
----------------------
There is also a corrections server that can be used as a datasource for corrections.
To use it you will need an http client initializaed with the correction URL and an access token header.

.. code-block:: python

    import rframe
    import straxen

    datasource = rframe.BaseHttpClient(URL,
                                       headers={"Authorization": "Bearer: TOKEN"})
    
    gain_docs = straxen.PmtGains.find(datasource, pmt=1, version='v3')


the easiest way to use the server is from the xeauth package, just run `pip install xeauth`

.. code-block:: python

    import xeauth
    import straxen

    datasources = xeauth.cmt_login()
    # The script will attempt to open a browser for authentication
    # if the broswer does not open automatically, follow the link printed out.
    # Once you are authenticated as a xenon member, an access token will be
    # retrieved automatically.

    gains_datasource = datasources.pmt_gains
    # or
    gains_datasource = datasources['pmt_gains']

    gain_docs = straxen.PmtGains.find(gains_datasource, pmt=1, version='v3')


Inserting Corrections
---------------------

New correction documents can be inserted into a datasource with the `doc.save(datasource)` method.
Example:

.. code-block:: python

    import straxen

    doc = straxen.PmtGains(pmt=1, version='v3', value=1, ...)
    doc.save(datasource)

If all the conditions for insertion are met, e.g. the values for the given index not already being set, the insertion will be successful.

Of course you must have write access to the datasource for any insertion to succeed. The default datasources are all read-only.
When using the server to write values you must request a token with write permissions:

.. code-block:: python

    import xeauth
    import straxen

    # If you have to correction roles defined (correction expert), you can request a token with
    # extended scope i.e. write:all. This token will allow you to write to all correction collections
    # If you do not have the proper permissions, you will just get back the default token scope of read:all
    datasources = xeauth.cmt_login(scope='write:all')

    datasource = datasources['pmt_gains']

    doc = straxen.PmtGains(pmt=1, version='v3', value=1, ...)
    doc.save(datasource)


Overriding default datasources
------------------------------
You can change which datasource is used by default (for the current session) for a given correction in the correction_settings:

.. code-block:: python

    import straxen

    straxen.corrections_settings.datasources['pmt_gains'] = MY_DEFAULT_DATASOURCE
