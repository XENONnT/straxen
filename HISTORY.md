3.0.3 / 2025-01-25
-------------------
* Rename "algo" to "alg" because they were mixed by @dachengx in https://github.com/XENONnT/straxen/pull/1523
* Bump version by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1500
* [DAQ] Bootstrax dynamic adapt targets for AmBe high rate by @cfuselli in https://github.com/XENONnT/straxen/pull/1492
* Track position reconstruction algorithm in `EventPatternFit` plugin by @dachengx in https://github.com/XENONnT/straxen/pull/1524
* Still use `np.bool_` in numba decorated function by @dachengx in https://github.com/XENONnT/straxen/pull/1527
* First and last channel inside peak(let)s by @dachengx in https://github.com/XENONnT/straxen/pull/1525
* Handle the case where no hitlets are fully contained in peaklets by @dachengx in https://github.com/XENONnT/straxen/pull/1529
* Minor replacement `np.argsort` to `strax.stable_argsort` by @dachengx in https://github.com/XENONnT/straxen/pull/1530
* Automatically copy SOM dtype to peaks by @dachengx in https://github.com/XENONnT/straxen/pull/1531
* More flexible input of `find_n_competing` function by @dachengx in https://github.com/XENONnT/straxen/pull/1532
* Fix a numerical error bug in `events_nv` by @dachengx in https://github.com/XENONnT/straxen/pull/1534
* Skip `test_useless_frontend` if local RSE is not found by @dachengx in https://github.com/XENONnT/straxen/pull/1537
* Area in CNF position contour by @dachengx in https://github.com/XENONnT/straxen/pull/1538
* Remove CMT URLs by @jmosbacher in https://github.com/XENONnT/straxen/pull/1235
* Change dark rate monitoring of the software trigger to seconds by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1535
* Remove CMT by @dachengx in https://github.com/XENONnT/straxen/pull/1539
* Replace `10**9` by `straxen.units.s` by @dachengx in https://github.com/XENONnT/straxen/pull/1540
* Drop python 3.9 by @dachengx in https://github.com/XENONnT/straxen/pull/1541
* CNF posrec urlconfig fix by @juehang in https://github.com/XENONnT/straxen/pull/1519
* Bump xedocs version to v0.2.36 by @dachengx in https://github.com/XENONnT/straxen/pull/1543

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v3.0.2...v3.0.3


3.0.2 / 2025-01-13
-------------------
* Collect SOM dtype at one place by @dachengx in https://github.com/XENONnT/straxen/pull/1511
* Stop support for list of "take" protocol by @dachengx in https://github.com/XENONnT/straxen/pull/1517
* Add `stage` flag for `RucioRemoteBackend` by @dachengx in https://github.com/XENONnT/straxen/pull/1520

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v3.0.1...v3.0.2


3.0.1 / 2024-12-27
-------------------
* Fix run_doc for led plugin by @GiovanniVolta in https://github.com/XENONnT/straxen/pull/1462
* Check RSE in `_find` method of `RucioRemoteFrontend` by @dachengx in https://github.com/XENONnT/straxen/pull/1464
* Garbage collection after calculated each chunk in `peak_positions_mlp` by @dachengx in https://github.com/XENONnT/straxen/pull/1467
* Enforce stable sorting in `np.sort` and `np.argsort` by @dachengx in https://github.com/XENONnT/straxen/pull/1468
* Clean `deprecate_kwarg` by @dachengx in https://github.com/XENONnT/straxen/pull/1470
* Update strax version to v2.0.1 by @dachengx in https://github.com/XENONnT/straxen/pull/1473
* Remove expedients plugins because SOM will be default by @dachengx in https://github.com/XENONnT/straxen/pull/1472
* Remove 1T related codes by @dachengx in https://github.com/XENONnT/straxen/pull/1476
* Use SOM peaklets classification by default by @dachengx in https://github.com/XENONnT/straxen/pull/1471
* Fix theta uncertainty bug by @napoliion in https://github.com/XENONnT/straxen/pull/1466
* Remove URLConfig warning about sorting by @dachengx in https://github.com/XENONnT/straxen/pull/1477
* Merge branch 'sr1_leftovers' into master by @dachengx in https://github.com/XENONnT/straxen/pull/1478
* Fix small bug in CNF by @dachengx in https://github.com/XENONnT/straxen/pull/1479
* Remove GCN & CNN S2 pos-rec by @dachengx in https://github.com/XENONnT/straxen/pull/1484
* Set CNF as the default S2 (x, y) position-reconstruction by @dachengx in https://github.com/XENONnT/straxen/pull/1486
* Prototype of peaklets-level (x, y) S2 position reconstruction by @dachengx in https://github.com/XENONnT/straxen/pull/1482
* Rename old `PeakletClassification` as `PeakletClassificationVanilla` by @dachengx in https://github.com/XENONnT/straxen/pull/1487
* Remove Bayes models by @dachengx in https://github.com/XENONnT/straxen/pull/1488
* Rename `defualt_run_comments` -> `default_run_comments` by @dachengx in https://github.com/XENONnT/straxen/pull/1489
* Accelerate Euclidean distance by numba by @dachengx in https://github.com/XENONnT/straxen/pull/1493
* Move `set_nan_defaults` to be a stand-alone function by @dachengx in https://github.com/XENONnT/straxen/pull/1497
* Set CNF as the default S2 (x, y) position-reconstruction by @dachengx in https://github.com/XENONnT/straxen/pull/1494
* Back to fixed window in LED calibration by @GiovanniVolta in https://github.com/XENONnT/straxen/pull/1499
* Move `compute_center_times` from straxen to strax by @dachengx in https://github.com/XENONnT/straxen/pull/1501
* Use numpy and strax native dtypes, not `"<i8"` or `"<f4"` by @dachengx in https://github.com/XENONnT/straxen/pull/1502
* Inherit `area_fraction_top`, `center_time` and `median_time` from peaklets by @dachengx in https://github.com/XENONnT/straxen/pull/1503
* Bump version of changed plugins in #1503 by @dachengx in https://github.com/XENONnT/straxen/pull/1504
* Clean unnecessary codes by @dachengx in https://github.com/XENONnT/straxen/pull/1507
* Clean chunk after computing `records` by @dachengx in https://github.com/XENONnT/straxen/pull/1508
* Add a line of comment about memory optimization by @dachengx in https://github.com/XENONnT/straxen/pull/1509

New Contributors
* @napoliion made their first contribution in https://github.com/XENONnT/straxen/pull/1466

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v3.0.0...v3.0.1


3.0.0 / 2024-10-24
-------------------
* Inherit `DEFAULT_CHUNK_SPLIT_NS` from strax by @dachengx in https://github.com/XENONnT/straxen/pull/1405
* Use `pyproject.toml` to install straxen by @dachengx in https://github.com/XENONnT/straxen/pull/1408
* Be compatible with new `Plugin.run_id` by @dachengx in https://github.com/XENONnT/straxen/pull/1410
* Small restrax fix - DAQ by @cfuselli in https://github.com/XENONnT/straxen/pull/1402
* Make targeted `raw_records` chunk 500MB by @dachengx in https://github.com/XENONnT/straxen/pull/1412
* Bump actions/setup-python from 5.1.0 to 5.1.1 by @dependabot in https://github.com/XENONnT/straxen/pull/1403
* fix peak per event plugin by @RoiFrankel in https://github.com/XENONnT/straxen/pull/1400
* Dynamic led window by @tflehmke in https://github.com/XENONnT/straxen/pull/1401
* Plugins for position reconstruction with conditional normalizing flow by @juehang in https://github.com/XENONnT/straxen/pull/1404
* Remove redundant spaces by @dachengx in https://github.com/XENONnT/straxen/pull/1411
* Stop using `self.config` because we do not use `strax.Option` by @dachengx in https://github.com/XENONnT/straxen/pull/1413
* Remove `DetectorSynchronization` by @dachengx in https://github.com/XENONnT/straxen/pull/1414
* Remove configuration `sum_waveform_top_array` from `MergedS2s` by @dachengx in https://github.com/XENONnT/straxen/pull/1415
* Debug for `EventwBayesClass` because peaks overlapping by @dachengx in https://github.com/XENONnT/straxen/pull/1417
* Refactor nv plugins by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1228
* Changed NV software trigger by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1388
* Assign `__version__` of `RecordsFromPax` by @dachengx in https://github.com/XENONnT/straxen/pull/1418
* Minor debug for the `pyproject.toml` by @dachengx in https://github.com/XENONnT/straxen/pull/1420
* Fix the usage of scripts by @dachengx in https://github.com/XENONnT/straxen/pull/1423
* Deprecate selection_str by @dachengx in https://github.com/XENONnT/straxen/pull/1424
* Fix singleton pattern for `MongoDownloader` by @dachengx in https://github.com/XENONnT/straxen/pull/1426
* Use more `strax.RUN_METADATA_PATTERN` by @dachengx in https://github.com/XENONnT/straxen/pull/1432
* Be compatible with utilix>0.9 by @dachengx in https://github.com/XENONnT/straxen/pull/1433
* Specify available RSE in `RucioRemoteBackend` by @dachengx in https://github.com/XENONnT/straxen/pull/1435
* Add docstring rucio by @yuema137 in https://github.com/XENONnT/straxen/pull/1436
* `pymongo_collection` is a bit confusing by @dachengx in https://github.com/XENONnT/straxen/pull/1437
* Put the RunDB API interface and MongoDB interface together by @yuema137 in https://github.com/XENONnT/straxen/pull/1442
* change N_chunk to URLConfig in peak_positions_cnf by @juehang in https://github.com/XENONnT/straxen/pull/1443
* Bump actions/setup-python from 5.1.1 to 5.2.0 by @dependabot in https://github.com/XENONnT/straxen/pull/1419
* Fixed default window position by @tflehmke in https://github.com/XENONnT/straxen/pull/1429
* Add `data_start` to temporary dtype for `events_nv` by @dachengx in https://github.com/XENONnT/straxen/pull/1447
* Add level in the tree when drawing dependency tree by @dachengx in https://github.com/XENONnT/straxen/pull/1446
* Move the whole mongo_storage module to utilix by @yuema137 in https://github.com/XENONnT/straxen/pull/1445
* Following the breaking change of https://github.com/AxFoundation/strax/pull/910 by @dachengx in https://github.com/XENONnT/straxen/pull/1452
* Switch to master for docformatter by @yuema137 in https://github.com/XENONnT/straxen/pull/1453
* Adjust saving preference by @yuema137 in https://github.com/XENONnT/straxen/pull/1451
* Save first samples of peak(lets) waveform by @HenningSE in https://github.com/XENONnT/straxen/pull/1406
* Only use `ThreadedMailboxProcessor` when `allow_multiprocess=True` by @dachengx in https://github.com/XENONnT/straxen/pull/1455
* Remove redundant pos recon by @yuema137 in https://github.com/XENONnT/straxen/pull/1449
* Clean `DeprecationWarning` and simplify plugins by @dachengx in https://github.com/XENONnT/straxen/pull/1456
* Update  dependencies of strax, remove git repo from dependency list by @dachengx in https://github.com/XENONnT/straxen/pull/1458
* Add fix integration window for the noise runs by @GiovanniVolta in https://github.com/XENONnT/straxen/pull/1457
* Use `max_time` when calculating peaklets properties by @dachengx in https://github.com/XENONnT/straxen/pull/1459

New Contributors
* @RoiFrankel made their first contribution in https://github.com/XENONnT/straxen/pull/1400
* @tflehmke made their first contribution in https://github.com/XENONnT/straxen/pull/1401
* @juehang made their first contribution in https://github.com/XENONnT/straxen/pull/1404
* @HenningSE made their first contribution in https://github.com/XENONnT/straxen/pull/1406

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.2.5...v3.0.0


2.2.5 / 2024-08-17
-------------------
* Generate only one instance for `MongoDownloader` by @dachengx in https://github.com/XENONnT/straxen/pull/1398
* Load whole run for `VetoIntervals` regardless the run length by @dachengx in https://github.com/XENONnT/straxen/pull/1399

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.2.4...v2.2.5


2.2.4 / 2024-07-01
-------------------
* Parse USERDISK base on hostname in RunDB by @dachengx in https://github.com/XENONnT/straxen/pull/1384
* Fix packages temporarily for documentation generation by @dachengx in https://github.com/XENONnT/straxen/pull/1385
* Bad url warnings by @LuisSanchez25 in https://github.com/XENONnT/straxen/pull/1216
* Allow local blinding files by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1387
* Lock strax version in test by @dachengx in https://github.com/XENONnT/straxen/pull/1389
* Add xedocs version to context config, only if xedocs is called by @Ananthu-Ravindran in https://github.com/XENONnT/straxen/pull/1393
* Revert "Lock strax version in test" by @dachengx in https://github.com/XENONnT/straxen/pull/1394

New Contributors
* @Ananthu-Ravindran made their first contribution in https://github.com/XENONnT/straxen/pull/1393

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.2.3...v2.2.4


2.2.3 / 2024-05-16
-------------------
* No need to set `loop_over` for `EventBasics` by @dachengx in https://github.com/XENONnT/straxen/pull/1377
* Initialize plugins whose `depends_on` is property by @dachengx in https://github.com/XENONnT/straxen/pull/1379
* Collect functions used for documentation building in `docs_utils.py` by @dachengx in https://github.com/XENONnT/straxen/pull/1380

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.2.2...v2.2.3


2.2.2 / 2024-04-30
-------------------
* Minor change of indents by @dachengx in https://github.com/XENONnT/straxen/pull/1341
* Remove unused `__all__` by @dachengx in https://github.com/XENONnT/straxen/pull/1342
* Bump graphviz from 0.20.1 to 0.20.2 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1345
* Specifically install `lxml_html_clean` by @dachengx in https://github.com/XENONnT/straxen/pull/1352
* Improve InterpolateAndExtrapolate performance for array valued maps by @l-althueser in https://github.com/XENONnT/straxen/pull/1347
* Bump graphviz from 0.20.2 to 0.20.3 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1350
* Bump actions/setup-python from 5.0.0 to 5.1.0 by @dependabot in https://github.com/XENONnT/straxen/pull/1351
* Add `storage_graph` to show the plugins stored or needed to be calculated in the dependency tree by @dachengx in https://github.com/XENONnT/straxen/pull/1353
* Small bug fix of `storage_graph`, save plot into desired folder by @dachengx in https://github.com/XENONnT/straxen/pull/1356
* Check non-positive lone_hits by @dachengx in https://github.com/XENONnT/straxen/pull/1358
* Return the edge closer to the target in `_numeric_derivative` by @dachengx in https://github.com/XENONnT/straxen/pull/1355
* Add a simply function to plot the dependency tree by @dachengx in https://github.com/XENONnT/straxen/pull/1363
* Remove `PeakSubtyping` from straxen by @dachengx in https://github.com/XENONnT/straxen/pull/1365
* Remove `xnt_simulation_config` by @dachengx in https://github.com/XENONnT/straxen/pull/1366
* Tolerate more exceptions when can not import admix by @dachengx in https://github.com/XENONnT/straxen/pull/1367
* Add `PeakSEDensity` and `EventSEDensity` by @dachengx in https://github.com/XENONnT/straxen/pull/1368
* Update `se_time_search_window_left` by @dachengx in https://github.com/XENONnT/straxen/pull/1370
* remove resource_cache from dali by @yuema137 in https://github.com/XENONnT/straxen/pull/1372
* Add `exclude_pattern` argument to `dependency_tree` by @dachengx in https://github.com/XENONnT/straxen/pull/1373
* Let xedocs to handle avg seg and seg partitioning by @GiovanniVolta in https://github.com/XENONnT/straxen/pull/1371

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.2.1...v2.2.2


2.2.1 / 2024-02-21
-------------------
* Loosen `save_when` of `Events` by @dachengx in https://github.com/XENONnT/straxen/pull/1327
* Deprecate the usage of `XENONnT/ax_env` by @dachengx in https://github.com/XENONnT/straxen/pull/1329
* `_text_formats` should include txt but not text by @dachengx in https://github.com/XENONnT/straxen/pull/1324
* Fix numerical comparison error of `test_patternfit_stats` by @dachengx in https://github.com/XENONnT/straxen/pull/1334
* Remove some packages requirements from `requirements-tests.txt` by @dachengx in https://github.com/XENONnT/straxen/pull/1337
* Fixing hitlets boundary out of chunk by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1328

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.2.0...v2.2.1


2.2.0 / 2024-01-16
-------------------
* remove cnn in s2_recon_pos_diff by @ZhenhaoLiangW in https://github.com/XENONnT/straxen/pull/1313
* Update pymongo version by @dachengx in https://github.com/XENONnT/straxen/pull/1316
* Use `straxen.EventBasics.set_nan_defaults` to set default values by @dachengx in https://github.com/XENONnT/straxen/pull/1317
* Update to bokeh v3 and holoviews v1, drop py3.8 support by @dachengx in https://github.com/XENONnT/straxen/pull/1318
* Drop 3.11 support for now by @dachengx in https://github.com/XENONnT/straxen/pull/1321
* Move all simulation contexts to WFSim by @dachengx in https://github.com/XENONnT/straxen/pull/1320
* Add nopython by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1319

New Contributors
* @ZhenhaoLiangW made their first contribution in https://github.com/XENONnT/straxen/pull/1313

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.1.6...v2.2.0


2.1.6 / 2023-12-21
-------------------
* Update bootstrax strax logging by @cfuselli in https://github.com/XENONnT/straxen/pull/1252
* Hint unstaged repo as well for `print_versions` by @dachengx in https://github.com/XENONnT/straxen/pull/1288
* Proposal to use pre-commit for continuous integration by @dachengx in https://github.com/XENONnT/straxen/pull/1240
* Update README by remove code style checking, add more ignore commits by @dachengx in https://github.com/XENONnT/straxen/pull/1290
* Check by default basics by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1287
* Use pre-commit for continuous integration also for scripts by @dachengx in https://github.com/XENONnT/straxen/pull/1293
* Find time difference and properties of nearest triggering peaks by @dachengx in https://github.com/XENONnT/straxen/pull/1301
* Update NaN filtering in InterpolatingMap by @JelleAalbers in https://github.com/XENONnT/straxen/pull/1302
* change integration window by @marianarajado in https://github.com/XENONnT/straxen/pull/1303
* Pull out FakeDAQ to legacy plugins by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1292
* Add gps plugins by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1285
* Make peaklets dtype flexiable by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1299
* add kwargs to simulation context by @LuisSanchez25 in https://github.com/XENONnT/straxen/pull/1277
* Fix photoionization correction to conserve cS2 by @xzh19980906 in https://github.com/XENONnT/straxen/pull/1306
* add_ref_mon_nv_plugin by @eangelino in https://github.com/XENONnT/straxen/pull/1307
* Update som classifcation by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1300
* Move ref mon to online and add to bootstrax by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1308
* Update corrected_areas.py by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1310

New Contributors
* @marianarajado made their first contribution in https://github.com/XENONnT/straxen/pull/1303
* @eangelino made their first contribution in https://github.com/XENONnT/straxen/pull/1307

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.1.5...v2.1.6


2.1.5 / 2023-10-11
-------------------
* Som plugin by @LuisSanchez25 in https://github.com/XENONnT/straxen/pull/1269


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.1.4...v2.1.5


2.1.4 / 2023-10-04
-------------------
* No need to apply `strax.check_chunk_n` individually by @dachengx in https://github.com/XENONnT/straxen/pull/1267
* Bump xedocs from 0.2.24 to 0.2.25 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1265
* Bump wfsim from 1.0.2 to 1.1.0 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1264
* Update configuration of RTD, add xedocs docs by @dachengx in https://github.com/XENONnT/straxen/pull/1271
* Add pad-array protocol by @jmosbacher in https://github.com/XENONnT/straxen/pull/1266


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.1.3...v2.1.4

2.1.3 / 2023-09-08
-------------------
* Manually check event overlapping by @FaroutYLq in https://github.com/XENONnT/straxen/pull/1214
* Add missing docstrings by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1234
* Use formatted float in `dataframe_to_wiki` by @dachengx in https://github.com/XENONnT/straxen/pull/1231
* Bump actions/setup-python, urllib3 and sphinx by @dachengx in https://github.com/XENONnT/straxen/pull/1232
* Update module index of docs by @dachengx in https://github.com/XENONnT/straxen/pull/1233
* Bump sphinx-rtd-theme from 1.2.2 to 1.3.0 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1238
* Replace `z` from `z_naive` to `z_dv_corr` by @dachengx in https://github.com/XENONnT/straxen/pull/1239
* Remove context collection badge by @dachengx in https://github.com/XENONnT/straxen/pull/1241
* Update xedocs version by @dachengx in https://github.com/XENONnT/straxen/pull/1246
* No need to get map shape for 0D placeholder map by @dachengx in https://github.com/XENONnT/straxen/pull/1245
* Give `RunDB` an option to find files in storage by @dachengx in https://github.com/XENONnT/straxen/pull/1244
* Check chunk n for backends after chunk loading by @dachengx in https://github.com/XENONnT/straxen/pull/1243
* Revert "Give RunDB an option to find files in storage but not in data… by @dachengx in https://github.com/XENONnT/straxen/pull/1248
* Photon ionization correction on S2 by @xzh19980906 in https://github.com/XENONnT/straxen/pull/1247
* Bump xedocs from 0.2.23 to 0.2.24 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1250
* FDC uses corrected position by @shenyangshi in https://github.com/XENONnT/straxen/pull/1254
* Correct elife at the last in `corrected_areas` by @dachengx in https://github.com/XENONnT/straxen/pull/1258
* Correct elife for `cs2_wo_timecorr` by @dachengx in https://github.com/XENONnT/straxen/pull/1260
* SR1 offline simulation strax context by @shenyangshi in https://github.com/XENONnT/straxen/pull/1253

New Contributors
* @xzh19980906 made their first contribution in https://github.com/XENONnT/straxen/pull/1247

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.1.2...v2.1.3


2.1.2 / 2023-07-28
-------------------
* Validate final type after URL eval by @jmosbacher in https://github.com/XENONnT/straxen/pull/1217
* Fix URLConfig.evaluate_dry by @jmosbacher in https://github.com/XENONnT/straxen/pull/1219
* Add function to save itp_map InterpolatingMap related dictionary into pickle by @dachengx in https://github.com/XENONnT/straxen/pull/1221
* Rename `tf_peak_model_s1_cnn` to `tf_model_s1_cnn` by @dachengx in https://github.com/XENONnT/straxen/pull/1223


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.1.1...v2.1.2


2.1.1 / 2023-07-06
-------------------
* Fix timing of peaks when ordering in `center_time` by @dachengx in https://github.com/XENONnT/straxen/pull/1208
* Move `get_window_size` factor of merged_s2s as untracked configuration by @dachengx in https://github.com/XENONnT/straxen/pull/1209
* Sort `hitlets` in `nVETOHitlets` by @dachengx in https://github.com/XENONnT/straxen/pull/1210
* Only print out warning once by @LuisSanchez25 in https://github.com/XENONnT/straxen/pull/1211


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.1.0...v2.1.1


2.1.0 / 2023-06-22
-------------------
* Added peaks subtyping by @Jianyu010 in https://github.com/XENONnT/straxen/pull/1152
* Fix ipython version by @dachengx in https://github.com/XENONnT/straxen/pull/1169
* Fix bug in hitlets time ordering by @dachengx in https://github.com/XENONnT/straxen/pull/1173
* Bump actions/setup-python from 4.5.0 to 4.6.0 by @dependabot in https://github.com/XENONnT/straxen/pull/1170
* Save hits level information(hits height and time difference) in peak and event level by @dachengx in https://github.com/XENONnT/straxen/pull/1155
* Fix argsort inside numba.jit using kind='mergesort' by @dachengx in https://github.com/XENONnT/straxen/pull/1176
* Bump merged_s2s version following `strax.merge_peaks` by @dachengx in https://github.com/XENONnT/straxen/pull/1179
* Use same files names for peak and event level pos-rec by @dachengx in https://github.com/XENONnT/straxen/pull/1160
* Update multi scatter Ignore nan in the sum of peaks. by @michaweiss89 in https://github.com/XENONnT/straxen/pull/1162
* Add dynamic event display docs by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1077
* Lower the titles in the same notebook by @dachengx in https://github.com/XENONnT/straxen/pull/1183
* No longer test `st.runs` in `test_extract_latest_comment_lone_hits` by @dachengx in https://github.com/XENONnT/straxen/pull/1199
* Remove unnecessary check in `merged_s2s` by @dachengx in https://github.com/XENONnT/straxen/pull/1195
* automatically appending local rucio path by @FaroutYLq in https://github.com/XENONnT/straxen/pull/1182
* Performance boost veto proximity by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1181
* Update build_datastructure_doc.py by @PeterGy in https://github.com/XENONnT/straxen/pull/1202
* Add rundoc URLConfig protocol by @jmosbacher in https://github.com/XENONnT/straxen/pull/1135
* Split event_area_per_channel into two plugins: event_area_per_channel… by @minzhong98 in https://github.com/XENONnT/straxen/pull/1191
* Fix event basics time ordering by @jjakob03 in https://github.com/XENONnT/straxen/pull/1194
* Make apply_xedocs_configs more flexible by @jmosbacher in https://github.com/XENONnT/straxen/pull/1204
* Try to make hashing more coinsistent by @LuisSanchez25 in https://github.com/XENONnT/straxen/pull/1201

New Contributors
* @PeterGy made their first contribution in https://github.com/XENONnT/straxen/pull/1202
* @minzhong98 made their first contribution in https://github.com/XENONnT/straxen/pull/1191

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.7...v2.1.0


2.0.7 / 2023-04-25
-------------------
* Bootstrax target removal after failures by @cfuselli in https://github.com/XENONnT/straxen/pull/1145
* reforming _raw_path and _processed_path by @FaroutYLq in https://github.com/XENONnT/straxen/pull/1149
* Adding correction of Z position due to non-uniform drift velocity by @terliuk in https://github.com/XENONnT/straxen/pull/1148
* Bump the versions of peaklets and quality check runs-on by @dachengx in https://github.com/XENONnT/straxen/pull/1153
* S1-Based 3D Position Reconstruction by @matteoguida in https://github.com/XENONnT/straxen/pull/1146
* Bump xedocs from 0.2.14 to 0.2.16 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1158
* Use zstd as compressor of peaks by @dachengx in https://github.com/XENONnT/straxen/pull/1154
* Bump sphinx from 5.3.0 to 6.2.0 in /extra_requirements by @dependabot in https://github.com/XENONnT/straxen/pull/1161

New Contributors
* @cfuselli made their first contribution in https://github.com/XENONnT/straxen/pull/1145
* @matteoguida made their first contribution in https://github.com/XENONnT/straxen/pull/1146
* @hmdyt made their first contribution in https://github.com/XENONnT/straxen/pull/1159

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.6...v2.0.7


2.0.6 / 2023-03-08
-------------------
* Bump supercharge/mongodb-github-action from 1.8.0 to 1.9.0 by @dependabot in https://github.com/XENONnT/straxen/pull/1140
* Small patches to restrax module by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1143, d04a3428c52c159577b61af2a28ddd0af5652027, 602b807291211f083c8f54df6768b8198fbf6b55
* Ms events by @michaweiss89 and @HenningSE in https://github.com/XENONnT/straxen/pull/1080

New Contributors
* @michaweiss89 made their first contribution in https://github.com/XENONnT/straxen/pull/1080

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.5...v2.0.6

Notes:
 - new data types: `peaks_per_event`, `event_top_bottom_params`, `peaks_corrections` (see #1080)


2.0.5 / 2023-02-24
-------------------
* fix xedocs for testing by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1139
* Restart python style guide by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1138
* Decrease number of chunks by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1123
* Restrax by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1074


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.4...v2.0.5


2.0.4 / 2023-01-16
-------------------
* Top and bottom timing parameters at event and peak level by @terliuk in https://github.com/XENONnT/straxen/pull/1119
* Allow use of xedocs context configs by @jmosbacher in https://github.com/XENONnT/straxen/pull/1125
* Bump actions/setup-python from 4.3.0 to 4.4.0 by @dependabot in https://github.com/XENONnT/straxen/pull/1128
* Add entry points by @jmosbacher in https://github.com/XENONnT/straxen/pull/1120
* URLConfig preprocessor by @jmosbacher in https://github.com/XENONnT/straxen/pull/1110
* Fix bootstrax timeouts by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1133


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.3...v2.1.0

Notes:
 - new data types: `peak_top_bottom_params`, `event_top_bottom_params`

2.0.3 / 2022-11-09
-------------------
* Adding peak waveforms at event level by @terliuk in https://github.com/XENONnT/straxen/pull/1112


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.2...v2.0.3

Notes:
 * lineage changes for event_area_per_channel


2.0.2 / 2022-10-24
-------------------
* New URLConfig protocols - list-to-array and list-to-dict by @LuisSanchez25 in https://github.com/XENONnT/straxen/pull/1104
* Single core 1T test by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1109

New Contributors
* @LuisSanchez25 made their first contribution in https://github.com/XENONnT/straxen/pull/1104


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.1...v2.0.2


2.0.1 / 2022-10-20
-------------------
* Use mongodb v4.4.1 when testing to match real version used in production by @jmosbacher in https://github.com/XENONnT/straxen/pull/1103
* Pass tests from remote forks by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1105
* Local minimum info 2 by @JYangQi00 in https://github.com/XENONnT/straxen/pull/1106
* Don't test without `strax.processor.SHMExecutor` by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1107
* Lower the default config value of online_max_bytes by @mflierm in https://github.com/XENONnT/straxen/pull/1108


New Contributors
* @JYangQi00 made their first contribution in https://github.com/XENONnT/straxen/pull/1106


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v2.0.0...v2.0.1


2.0.0 / 2022-10-17
-------------------
* Fix acqmon veto field by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1072
* Use self.dtype also for empty peaks by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1058
* Re Start style guide by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1084
* Transition plugins to URLConfig by @jmosbacher in https://github.com/XENONnT/straxen/pull/1079
* Fix help of peak basics. by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1081
* Remove `tight_coincidence_channel` fix #1078 by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1092
* Add new `s1_pattern_map`, fix #1070 by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1093
* Restructure plugins by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1094
* Return on single delele by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1095
* Never change raw_records by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1096
* fix missing export by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1098
* Event level S2 posrec by @terliuk in https://github.com/XENONnT/straxen/pull/1097
* New tpc event display by @WenzDaniel in https://github.com/XENONnT/straxen/pull/1043
* Change timeouts by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1101
* Option to add top bottom wf by @petergaemers @DCichon @FaroutYLq @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1099

Notes:
 * Lineage changes for all data except raw-records due to #1079
 * Breaking changes induced in strax [v1.4.0](https://github.com/AxFoundation/strax/releases/tag/v1.4.0)
 * Changed signatures of plugins in [#1094](https://github.com/XENONnT/straxen/pull/1094)
 * New plugins for event level processing by [#1097](https://github.com/XENONnT/straxen/pull/1097)


**Full Changelog**: https://github.com/XENONnT/straxen/compare/v1.8.3...v2.0.0


1.8.3 / 2022-07-18
-------------------
* Bootstrax file-check fix by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1064
* Fix hanging straxer by @jmosbacher in https://github.com/XENONnT/straxen/pull/1065

Notes:
* No lineage changes

Full Changelog:
 - https://github.com/XENONnT/straxen/compare/v1.8.2...v1.8.3


1.8.2 / 2022-07-12
-------------------
* Stop tf pbar by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1063
* Allow long runs by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1062

Notes:
* No lineage changes

Full Changelog:
- https://github.com/XENONnT/straxen/compare/v1.8.1...v1.8.2


1.8.1 / 2022-06-07
-------------------
Minor:
* Change FDC z offset and add alternative interaction by @ftoschi in #1017
* Plugin for online individual peak monitoring by @mflierm in #1054

Notes:
* Lineage changes for event_positions, corrected_areas, energy_estimates, event_info, event_info_double
Added new data-kind: individual_peak_monitor

Patch:
* Version logging by @mflierm in #1055
* update docs ev interactive display by @JoranAngevaare in #1042
* allow dry eval of URL configs by @JoranAngevaare in #1040
* refactor tests by @JoranAngevaare in #1030
* start testing examples of notebooks by @JoranAngevaare in #1048
* Bump nbsphinx from 0.8.8 to 0.8.9 in /extra_requirements by @dependabot in #1053
* Add kicp to query by @JoranAngevaare in #1052
* Bump sphinx from 4.5.0 to 5.0.1 in /extra_requirements by @dependabot in #1051
* Allow constant tuple options by @JoranAngevaare in #1039

Full changelog:
- https://github.com/XENONnT/straxen/compare/v1.7.1...v1.8.0

1.7.2 / 2022-07-18
-------------------
Patch:
 * Upload cherry picks by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1066
 * Fix hanging straxer by @jmosbacher in https://github.com/XENONnT/straxen/pull/1065
 * Stop tf pbar by @JoranAngevaare in https://github.com/XENONnT/straxen/pull/1063
 * update docs ev interactive display by @JoranAngevaare in #1042
 * allow dry eval of URL configs by @JoranAngevaare in #1040
 * refactor tests by @JoranAngevaare in #1030
 * start testing examples of notebooks by @JoranAngevaare in #1048
 * Bump nbsphinx from 0.8.8 to 0.8.9 in /extra_requirements by @dependabot in #1053
 * Add kicp to query by @JoranAngevaare in #1052
 * Bump sphinx from 4.5.0 to 5.0.1 in /extra_requirements by @dependabot in #1051
 * Allow constant tuple options by @JoranAngevaare in #1039

**Full Changelog**: https://github.com/XENONnT/straxen/compare/v1.7.1...v1.7.2


1.7.1 / 2022-05-16
-------------------
Patch:
* Check if processed data already exists in --production mode by @mflierm in https://github.com/XENONnT/straxen/pull/1024


Notes:
- No lineage changes

Full Changelog:
 - https://github.com/XENONnT/straxen/compare/v1.7.0...v1.7.1

New Contributors
 - @mflierm made their first contribution in https://github.com/XENONnT/straxen/pull/1024


1.7.0 / 2022-05-11
---------------------
Minor:
- Fix detector sync (#1033)
- Numbafy function (#1015)
- Fixing binomial (#991)
- Patched wrong setting (#1014)
- Partitioned tpc (#1027)

Patch:
- Update requirements-tests.txt (#1021)
- remove deprecated function (#1023)
- Warn when context not from cutax (#1020)
- Add 'electron_diffusion_cte' variable from CMT (#1025)
- Start testing with PluginTestingSuite, fix #881 (#1022)
- add dict type correction (#1028)

Notes:
- Lineage changes for event_area_per_channel, event_pattern_fit, peak_classification_bayes, detector_time_offsets, event_sync_nv


Full Changelog:
 - https://github.com/XENONnT/straxen/compare/v1.6.2...v1.7.0


1.6.2 / 2022-05-06
---------------------
Patch:
-  Add MV trigger channel to acqmon hits https://github.com/XENONnT/straxen/pull/1035

Notes:
 - only lineage changes for dtypes > `aqmon_hits` (https://github.com/XENONnT/straxen/pull/1035)


Full Changelog:
 - https://github.com/XENONnT/straxen/compare/v1.6.1...v1.6.2


1.6.1 / 2022-04-12
------------------
Plugin fixes
- Remove records not hits. (#987)
- Remove Shadow&Ambience plugin SaveWhen.EXPLICIT (#982)
- fix issue 977 (#984)
- Position shadow sigma set to nan when S2 not positive (#980)
- Fix small bug if GPS has larger delay (#986)

Improved scripts / test
- iterative straxer targets (#973)
- Debug savewhen test (#963)
- Exit 0 on existing data in straxer (#970)
- dependabot remote (#1008)
- print util (#989)

Documentation
- Fix href datakind page (#969)

Storage fixes
- only find rucio from dali (#1010)
- Fix #1010 add midway as dali (#1012)
- Allow unused rucio local (#976)

Notes:
 - only lineage changes in `detector_time_offsets` ( #986)


1.6.0 / 2022-03-09
------------------
Minor:
- Patch corrected areas (#964)
- Patch in EventShadow (#967)
- Bump version of merged S2s (#919)
- Add Bayes peak probability (#954)
- fix hit sorting, add raw-record-aqm overlap check (#961)

Notes:
- Lineage changed for event_shadow, merged_s2s, corrected_area and aqmon_hits.
- Added new data_types peak_classification_bayes and event_w_bayes_class


1.5.0 / 2022-03-08
------------------
- Update Shadow plugins and add Ambience plugins (#912)
- Update online_monitor.py (#958)
- fix ref to PR in docs (#957)

Notes:
 - Lineage changes for online_monitor_mv
 - New plugins `peak_ambiance` and `event_ambiance` and changes in lineage of `event_shadow` and `peak_shadow`


1.4.0 / 2022-03-02
------------------
Minor:
- Update corrected areas (#931) and Update event_processing.py (#937)
- Fix bug veto positions (#940)
- S1 aft map & CMT tweaks (#916) and initialize plugin without setup() method (#953)

Patch
- Documentation building (#934)
- Development (#951)

Notes:
- Lineage changed for events_positions_nv, corrected_areas and event_pattern_fit due to option changes
- Lineage changes for events, event_basics, event_positions and peak_shadow due to change old config style to new URL style


1.3.0 / 2022-02-26
------------------
Minor:
- Rewrite aqmon processing (#929)
- Add plugin to compute time offsets per chunk (#917)
- Change alt_s2 identification criterion (#890)

Patch
- Remove median baseline from nveto processing (#927)
- Patch scada interface (#928)
- Updated nveto resolving time to 200 ns. Change to URL configs (#933)

Testing
- Enable holoviews testing py3.10 (#914)
- make coverage configuration (#926)

Notes:
- The lineage of the aqmon processing-chain changed
- The lineage of plugins > `event_basics` changed
- The lineages of the n/m-veto changed.


1.2.8 / 2022-02-16
------------------
Patch
- Remove veto tagging plugins from straxen (#918)
- Extend save when (#879)

Testing
- nestpy testing on py3.10 (#911)
- Simplify requirements (#913)
- Remove OM test that is collection status dependent (#921)
- Remove data after testing (#923)

Notes:
- Removed nveto/mveto tagging plugins (#918)
- Changed saving behavior of `pulse_counts` (#879)


1.2.7 / 2022-02-03
------------------
- (rucio)storage reorganization (#866)
- URLConfig documentation (#863)
- Fix leading zeros error (#889)
- Delete update context collection (#883)
- update github actions (#884)
- update print versions (#888)
- deprecate old python versions (#906)
- fix coveralls report (#905)
- merges from development branch (#910)

Notes:
 - no lineage changes


1.2.6 / 2022-01-18
------------------
fixes/tests:
- Fix online monitor test (#882)

notes:
- No lineage changes

1.2.5 / 2022-01-14
------------------
fixes/tests:
 - test with py3.10 (#878)
 - remove fixme error (e0e30d94ec8f5276c581da166787db72ba0eef4a)
 - bump numba (#880)
 - Tests for scada interface (#877)

notes:
 - No lineage changes

1.2.4 / 2022-01-10
------------------
fixes/tests:
 - Fixes for WFSim <-> CMT (#865)
 - Tests for WFSim contexts (#855)

notes:
 - First 1.2.X version compatible with WFSim
 - No lineage changes

1.2.3 / 2022-01-10
------------------
- Bump numpy (#876)

notes:
 - Incompatible with WFSim

1.2.2 / 2022-01-10
------------------
tests:
 - Test for Mongo-down/uploader (#859)
 - Test for rucio-documents in the rundb (#858)
 - Test for bokeh_utils (#857)
 - Tests for common.py fix #741 (#856)

bugfix:
 - Bump peaklets version (#873)

notes:
 - Lineage change for `peaklets` (#875)


1.2.1 / 2021-12-27
------------------
fixes/tests:
 - Add cmt tests and fix bug in apply_cmt_version (#860)
 - Pin documentation requirements (#862)
 - Add read the docs config (#861)
 - Pymongo requirement should be <4.0 (#852)

notes:
 - Bug for `peaklets-uhfusstvab` due to (#875)
 - No lineage changes
 - Incompatible with WFSim


1.2.0 / 2021-12-21
-------------------
major:

* Update CorrectedAreas (instead of EnergyEstimates) (#817)
* S2 pattern fit (#780)
* Exclude S1 as triggering peak (#779)
* Two manual boundaries (updated 11/24/2021) (#775)
* Add main peaks' shadow for event shadow (#770)
* Events synchronize (#761)
* Implement peak-level shadow and event-level shadow refactor (#753)
* use channel tight coincidence level (#745)

minor / patches:

* Normalized line endings (#833)
* Fix codefactor issues (#832)
* Another try at codefactor (#831)
* URLConfig take protocol for nested keys (#826)
* Rename tight coincidence (#825)
* Move URLConfig cache to global dictionary (#822)
* Remove codefactor (#818)
* Performance update for binomial test (#783)
* URLConfig not in strax (#781)
* Add refactor event building cut (#778)
* whipe online monitor data (#777)
* Cache dependencies (#772)
* Update definition array_valued (#757)

fixes/tests:

* Add test for filter_kwargs (#837)
* Fix nv testing data (#830)
* Unittest for DAQreader (#828)
* Fix broken matplotlib/minianalyses (#815)
* Itp test (#813)
* Loose packaging requirement (#810)
* can we disable codefactor please (#809)
* Fix #781 (#808)
* Matplotlib changed requirements (#805)
* Pin pymongo (#801)
* Bump wfsim tests (#773)
* Patch peaks merging (#767)

notes:
 - Bug for `peaklets-uhfusstvab` due to (#875)
 - plugins changed (new lineage) everything >= 'peaklet_classification'
 - offline CMT versions don't work in this release
 - Incompatible with WFSim


1.1.3 / 2021-11-19
-------------------
minor / patches:
- Add URL based configs (#758)
- Add perpendicular wires handling info and function (#756)
- Add a few special cases event_info_double (#740)
- Process afterpulses on ebs (#727)
- Add zenodo (#742)
- Set check_broken=False for RucioFrontend.find (#749)
- Explicitly set infer_dtype=False for all Options (#750)
- Use alt z for alternative s1 binomial test (#724)

fixes/tests:
- update docs (#743)
- Remove RuntimeError in RucioFrontend (#719)
- cleanup bootstrax logic for target determination (#768)
- Test installation without extra requirements (#725)
- Adding code comments for corrected z position (#763)
- Reactivate scada test (#764)
- Added resource exception for Scada (#755)
- test_widgets is broken? (#726)
- Track bokeh (#759)
- Fix keras requirement (#748)
- Update requirements-tests.txt (#739)
- Fix deprecation warning (#723)
- Update test_misc.py (90f2fc30141704158a0e297ea05679515a62b397)

notes:
 - plugins changed (new lineage) are `event_info_double` and `event_pattern_fit`


1.1.2 / 2021-10-27
-------------------
minor / patches:
- Plugin for afterpulse processing (#549)
- Veto online monitor (#707)
- Refactor straxen tests (#703)
- WFSim registry as argument for simulations context (#713)
- Update S1 AFT map in event pattern fit (#697)
- Refactor s2 correction (#704)

fixes/tests:
- Set default drift time as nan (#700)
- Revert auto inclusion of rucio remote #688 (#701)
- fix bug in CMT (#710)
- Fix one year querries (#711)
- Test new numba (#702)
- Unify CMT call in contexts (#717)
- Small codefactor patch (#714)
- test nv with nv data (#709)
- Add small test for wfsim (#716)

notes:
 - plugins changed (new lineage) are:
   - `afterpulses`
   - `online_monitor_nv`
   - `online_monitor_mv`
   - `event_pattern_fit`
   - `corrected_areas`

1.1.1 / 2021-10-19
-------------------
 - Fix to test for RunDB frontend when no test DB is sourced (6da2233)


1.1.0 / 2021-10-18
-------------------
major / minor:

- Previous S2 Shadow Plugin draft (#664)
- Use admix in straxen (#688)
- Add posdiff plugin (#669)
- updated S2 corrected area (#686)
- Version bump of hitlets (#690)
- Add n saturated channels (#691)
- add small tool to extract run comments from database (#692)
- Update online_monitor_nv to v0.0.3 (#696)


patches and fixes:

- Use read by index and check for NaNs (#661)
- Add small feature for printing versions of git (#665)
- Fix minianalyses from apply_selection (#666)
- fix some warnings from testing (#667)
- Add source to runs table (#673)
- Pbar patch for rundb query (#685)
- Implement SDSC as a local RSE for Expanse (#687)
- Skips superruns in rucio frontend (#689)
- Warn about non-loadable loggers (#693)
- Add RunDb read/write-test (#695)
- Fix bug in rucio frontend (#699)



1.0.0 / 2021-09-01
-------------------
major / minor:

- merge s2 without s1 (#645)
- First nVeto monitor plugin (#634)
- Peak event veto tagging (#618)
- Fix peaklet area bias (#601)
- Add lone hit information to merged S2s. (#623)


patches and fixes:

- Fix n_hits of peaks (#646)
- Update requirements for strax (#644)
- Modifications of nT simulation context (#602)
- Straxer for other packages (#595)
- [Bug fix] alt_s{i}_delay computation (#598)
- Bump version refactor code for cleanliness. (#597)
- Increase buffer size (#604)
- Stop testing py3.6 (#621)
- Remove online event monitor (#620)
- Add matplotlib to test requirements (#626)
- Fix rundb select runs with superruns (#627)
- Change EventInfo to save when explicit (#628)
- Update test data (#631)
- Allow database to not be initialized (#636)
- new plot_pmts (#637)
- Speed up event pattern fit (#625)
- kwargs for saver (#639)
- Add a plugin for external trigger run on nVeto calibration (#630)
- Fix veto event positions (#641)
- Use rucio from straxen & nest RucioRemote imports (#592)


0.19.3 / 2021-07-16
-------------------
- Rewrite EventBasics, set event level S1 tight coincidence (#569)
- New nt sim context & update get correction from CMT implementation (#555)
- Superruns (documentation) (#554, #594)

bootstrax / live processing
- Allow sub-mbs datarates and old runs (#572)
- increase input_timeout buffer daq reader (#593)
- Error logging bootstrax (#584)
- remove the id from the traceback (#585)

patches and fixes
- Reactivate scada tests (#583)
- Don't add test that you don't run - WFSim (#574)
- Fixing veto intervals time (#587)
- Patch scada interface (#588)
- reduce codefactor (#590)

0.19.2 / 2021-06-27
-------------------
- do not interpolate corrections if is an array (#570)

0.19.1 / 2021-06-24
-------------------
- Fix merged S2s upgrade #548 (#566, a2f5062, #568)
- Disable rucio frontend as default temporarily (#567)

0.19.0 / 2021-06-23 (bugged)
----------------------------
minor changes
- S1/S2 event patternfit and S1 AFT test (#499)
- Change tight_coincidence (#564)
- Fixing saturation correction bugs (#541)
- Rewrite merge s2 (#548)
- Compute width again after saturation correction (#542, #552)
- Add rucio frontend (#472, #553)
- Redo hit_thresholds (#543)
- Standardize CMT options as (correction, version, nT=boolean) (#457, #532)

patches and fixes:
- z coordinate update (#535)
- Fix example command (#547)
- Don't import holoviews and ipywidgets (#551)
- pre_apply_function from $HOME only in pytest (#559)
- Rundb should not crash on fuzzy (#561)
- Remove travis for testing from straxen (#557)
- Fix missing info in bootstrax docs, fix #546 (#558)
- Add scada interface to docs (#560)
- Tweaks for new release 0.19.0 (#562)


0.18.6-0.18.8 / 2021-06-03
-------------------
- Patches installation for pypi (#529, e880420, fce6d87)


0.18.5 / 2021-06-03
---------------------
- Allow variable event duration (#494, #528)
- Veto Proximity Plugin (#296)
- Apply database function prior to returning the data (#497)
- Max-size for rechunkable raw-records (#495)
- Itp map patch (#471)
- Bin updates (#502)
- Split requirement files, set autoupdate dependabot (#504)
- Fix failing tests (#503)
- Reduce review dog verbosity (#498)
- Reduce plugin testing time (#500)
- Patch remap cabled (#501)
- Fix veto veto regions (#493)


0.18.4 / 2021-05-20
---------------------
- Documentation and package maintenance (#481)
- Veto plugins (#465)
- Changed nveto splitting thresholds. (#490)
- Remove old unused contexts (#485)
- Use_per_run_defaults explicitly for 1T (#484)
- Set event_info_double as endpoint for kr (#480)
- Fix difference between datetime and date (#473)
- Fix _find for rucio to include transferred. Set kwarg defaults (#483)
- Fix AFT close but not quite 1 (#477)
- Fix online_monitor (#486)
- Activated overlapping check for mveto again. (#489)


0.18.3 / 2021-05-06
---------------------
- Update classifiers for pipy (#464)
- Fix for scan runs query (0cc47f2 )


0.18.2 / 2021-05-04
---------------------
- Nveto event display (#444)
- do check for overlaps in NV (#458)
- Refactor veto plugins (#463)
- Remove zero gain hits (#468)
- Time widget misc2 (#449)
- Added changes for user credentials (#392)
- Scada allowed, fix (#469)
- Added support of dill for resource files (#459)
- Reduce Pep8 gitHub bot verbosity (#466, #467)
- fix 1T sim context to have working dep. trees (#461)
- Reduced test complexity (#462)
- test python 3.9 (#407)
- fix keyerror for uploading data in selectruns (#455)


0.18.1 / 2021-04-23
---------------------
- Allow faster NV/MV by bootstrax (#440)
- Change records default processor (#441)
- Require data to be transferred to dali to load (#445)
- Wrap correction functions for mc optional config (#443)
- Use did for finding several runs (#451, 59afa35)
- Mveto events (#447)


0.18.0 / 2021-04-16
---------------------
- Clustering and classification update (#436)
- Documentation: add 1T, fix #31, compact config display (#434)
- Implement nT S1 correction (#430)
- Use CMT to get electron drift velocity (#437)
- Set max-runnumber (#433)
- Update update-context-collection.py (#438)
- Raise notimplemented error for peak_min_pmts > 2 (#432)
- Update apply_function_to_data (#431)
- use strax.apply_selection in om (#435)


0.17.0 / 2021-04-09
--------------------
- Extend event_basics and remove event_posrec_many (#420)
- Add nveto event tests (#425)
- Update veto_pulse_processing.py (#427)
- add option abbreviate event_display (#418)
- fix logic linked mode (#426)
- fix test to use tempdir (#423)
- Added output_notebook to data selector. (#421)
- bootstrax, fix abandonning (#422)


0.16.0 / 2021-04-02
--------------------
- add get_correction_from_cmt to corrections_services.py (#404, #409)
- Updated on the nveto plugins and new event plugins (#416, #389)
- New EventPositions for XENONnT (#396)
- Check for overlapping raw_records in nT sims (#413)
- Get n_veto gains from CMT (#406)
- Bug fix: Added fixed minimal length for temp_hitlets. #415
- use dependabot for actions (#414)
- Event display update, record matrix and dynamic data selector (#401)
- Remove duplicate call to 1T sim config (#411)
- Fix abandonning (#412)

0.15.9 / 2021-03-24
--------------------
- Use multiple targets for better online support (#397)
- Use code style commentator (#399, #405)
- Add daq-plots to minianalysies (#394)
- Check for raw-record overlaps veto systems (#390)
- Infer-target update (#395)
- Prevent abandonment of done runs (#398)
- Upload compressor from metadata to rundoc (#410)
- Cleanup ajax (#402)
- Context cleanup (#403)
- Change default nT fax config (#391)


0.15.8 / 2021-03-02
--------------------
- Update daq-tagging for abandoned runs (#374)
- Remove nones and replace with nans for itp map (#388)
- Check for raw-record overlaps (#387)


0.15.7 / 2021-02-26
--------------------
- Fix for commentjson-package for zipped json (#386)


0.15.6 / 2021-02-26
--------------------
- Scada updates (#378, #383)
- Correct S2(x,y) with CMT (#382)
- Correct elife with CMT (#385)
- Replace json with commentjson (#384)


0.15.5 / 2021-02-22
--------------------
- Patch version 0.15.3 (b5433bd)


0.15.3 / 2021-02-22
--------------------
- Test with database (#360)
- Fix issue #371 - alt s2 in event_posrec_many (#372)
- Update issue templates (#375)
- Link data structure to github page (#377)
- Fixes/improvements for 'plot_pulses' (#379)
- Remove unused code block (#380)


0.15.2 / 2021-02-17
--------------------
- GCN and CNN version bump for CMT bugfix (#367)
- Veto compression updates (#365)
- Simulation context fixed gains (363)


0.15.1 / 2021-02-11
--------------------
- Change event extensions (#364)


0.15.0 / 2021-02-09
--------------------
- Datarate dependent compressor (#358)
- Reduce n-files/run (#343)
- PulseProcessing save_when = strax.SaveWhen.TARGET (#352)
- Online events monitor (#349)
- Changed nveto baseline length (#362)
- Use DAQ logger (#354)
- Small hit pattern plotting bugfix (#357)
- Allow dynamic copy of dtype (#361)


0.14.5 / 2021-01-29
--------------------
- Function for version printing (#344)
- Extending the event window (#345)
- Check for daq-reader processing threads (#347)
- Update create-utilix-config.sh (#348)


0.14.4 / 2021-01-22
--------------------
- Nveto changes (#319)
- travis test at pinned environments (#335)
- Maintance and fixes on Bootstrax and ajax (#337, 96a2858, 84fda21, b09ea49, 1e577d9, 59cfd7d, 46ad1a3, 968a1dc)
- Some fixes and changes for the passive event display + Plotting tests (#338, 1d1b5b2, 93c7e18, 331b543, 055aa55, 1ce04ff)
- Listen to utilix, remove depricated function from straxen #340


0.14.3 / 2021-01-15
--------------------
- EventBasics dtype should be ordered (8665256)


0.14.2 / 2021-01-15
--------------------
- Add MLP, CNN and GCN position reconstruction (#323, #331, #332)
- Matplotlib event display (#326)
- Bokeh interactive event display (#330)
- New tutorials and updated documentation (#322)
- Scada-interface updates (#321, #324)


0.14.1 / 2021-01-04
--------------------
- bootstrax updates (39685a7, d0c3537, 874646a, df6e13f, 33d9da1, 2dfce7e)


0.14.0 / 2020-12-21
--------------------
- Bump version PulseProcessing for baseline fix (#317)
- Lower peak_min_pmts to 2 for nT (#299)
- Allow flexible SHEV (#266)


0.13.1 / 2020-12-21
--------------------
- fix requirements for numpy (#318)


0.13.0 / 2020-12-16
--------------------
- New (configuration)file handling module (#311)
- Updated documentation for file loading (#311)
- MV & NV integration using CMT (#312)
- Improved database interactions Bootstrax (#313, #314)
- Add 1-coincidence option for NV (#316)

0.12.5 / 2020-12-09
--------------------
- Muveto (#287)
- fix lone hit cut for online monitor (#308)


0.12.4 / 2020-12-06
--------------------
- Add temporary context (#302)
- Scada interface updates (#297, #301)
- Waveform plotting in minianalyses (#172)
- Update online_monitor for lone hits (#294)
- Tests for time selection fix strax/345 and more (#298)
- Add more tests to straxen (#292)
- Pytest on github actions (#303)
- Add coveralls to straxen (#290)
- Use github actions to update context collection (#282)
- Update simulation contexts (#286, #300)
- Remove to_pe_constant from CMT (#283)
- Use utilix for client in CMT (#288)
- Update straxer (#285)
- Bootstrax updates (#289)


0.12.3 / 2020-11-14
--------------------
- bugfix in desaturation correction (#279)


0.12.1 / 2020-11-13
--------------------
- CMT tweak before launch: ffill ONLINE corrections (#275)


0.12.0 / 2020-11-13
--------------------
- DAQReader for 2ns digitizers (#270)
- Activate CMT for PMT gains (#271)
- Desaturation correction (#244)
- Rise time requirement change (#273)
- Replace xenon_sectrets by ini file (#163)


0.11.1 / 2020-11-05
--------------------
- Corrections management tool CMT (#220, #251)
- Add Online Monitor plugins (#257, #265, #267)
- Add Scada interface for slow control data (#217)
- Documentation-updates (#246, #248)
- Update Rucio frontend (#254)
- Several (bug)fixes (#253, #262, #256)


0.11.0 / 2020-10-15
--------------------
- Separate context for fist commissioning data (#246)
- Online Monitor storage frontend (#216)
- Add Acquisition-monitor plugins (#207)
- Many (bug)fixes (#221, #223, #224, #226, #236, #238, #240, #241, #241, #245)
- Use CutPlugin class (#225)
- Bootstrax updates (#232)


0.10.1 / 2020-09-11
--------------------
- Added pytest to travis which builds data for all registered nT plugins (#193)
- Added moun- and neutron-veto into the streamline processing with bootstrax (#184)
- Added back-up URLs for the mongoDB (#213)
- Updated data cleaning/handling with ajax and bootstrax (#182, #191, #196, #202, #206)
- Updated documentation and doc-strings (#189, #192 and #198)
- Updated bin scripts like straxer (#204)
- Updated PMT gains (#208)
- Renamed high energy plugins (#200)
- Bugifx in nveto-plugins (#183, #209)
- Bugfix in clean_up_empty_records (#210)


0.10.0 / 2020-08-18
--------------------
- Neutron-veto integration (#86)
- Processing for high energy channels (#161, #176)
- Integrate rucio as storage backend (#164)
- Remapping of old runs (#166)
- Bootstrax/microstrax/ajax updates (#165)
- Pull request template (#168)
- Neural net for nT placeholder (#158)
- Forbid creation of any rr-type (#177)
- Add kwargs to 1T-contex (#167)
- Update LED-settings (#170)

0.9.2 / 2020-07-21
--------------------
- Change S1 split threshold (#150)
- Chunking endtimes in DAQReader (#146)
- Up version of peaklets for strax update (#152)
- Forbid users to create records (#153)
- Several updates in ajax, bootstrax and microstrax (#155, #151, #148, #147, #143)
- Bugfix led_calibration (#142)


0.9.1 / 2020-07-03
--------------------
- Rechunk pulse_counts and veto regions (#130)
- Add baseline info to pulse_counts (#140)
- Waveform plotting fixes (#137, #128)
- More gain model options (#132)
- Add ajax data removal script (#134)
- LED calibration update (#125)
- Bootstrax updates (#129)
- Update simulation context (#127)
- Fix n+1 bug in n_hits (#126)


0.9.0 / 2020-05-20
------------------
- Use revised coordinates for PMT positions (9da05b)
- Fix tutorials and holoviews display (32490b)
- Fix coordinate flipping in itp_map (#113)
- Fix n_hits field for peaklets (#122)
- Fix led_calibration options (#120)
- Fix n_top_pmts default (#119)
- Bootstrax updates (#112, #121)
- Update parameters for new rundb setup
- Specify immutabledict requirement (#116)


0.8.1 / 2020-05-05
------------------
- Update gains and hitfinder thresholds (#110)
- Fix cuts for strax v0.9.0 (#108)
- Bootstrax updates (#106, #109, #111)
- Fix peak_basics' max_pmt_area dtype (was int, is now float)
- Event scatter colorbar fix (#107)
- Fix tutorial notebook context names
- Add draw_box and dataframe_to_wiki

0.8.0 / 2020-04-28
-------------------
- Fix lone hit integration (#103, #105)
- Fix peak_right extension default (#104)
- Require 4 PMTs to make a peak for nT data (temporarily)
- Several bootstrax updates (#102, #100, #91, #90)
- Fix spurious free_options in xenon1t_led (#89)
- Add delay time computation to event_basics (#88)
- Update time end andtime for pulse_count (#85)


0.7.3 / 2020-04-13
-------------------
- Upgrade EventBasics (#65, #67, #68)
- Double scatter treemakers (#65, #67)
- Update pax converter for new strax (#87)
- Fix for LED processing (#84)
- Minor fixes for some warning messages


0.7.2 / 2020-04-06
------------------
- LED plugin and contexts (#42)
- Hitfinder fixes (adc65b, 5d1424)
- Bootstrax updates (#83, 93496b)
- Microstrax updates (855d18, 855d18)
- nT context / runDB updates (7cd138, 121e36)


0.7.1 / 2020-03-30
------------------
- Rename raw_records_lowgain to raw_records_he (#72)
- Fix n_tpc_pmts for nT (#73)
- Bootstrax updates (#74)
- microstrax to serve strax data as JSON over HTTP (#76)
- Update PMT plot for nT (#78)
- Fix: peaklets cannot extend past chunk boundaries (e63767)


0.7.0 / 2020-03-19
------------------
- DAQReader sorts out subdetector outputs (#64)
- Separate XENONnT and XENON1T contexts (#68)
- Start options for specifying gain model (#71)
- Auto-infer bootstrax processing settings (#70)


0.6.0 / 2020-03-05
------------------
- Updates for the new strax version (#60)
  - refresh_raw_records script to convert to new format
  - DAQReader creates artificial deadtime if needed to separation
  - PulseProcessing now baselines and flips the waveform
  - Software-HE veto buffer overrun fixes
  - Remove hacks for empty MergedS2 handling
  - Add time fields to all plugins
- Hitfinder update: noise- and channel-dependent thresholds (#55)
- PulseProcessing checks for overlaps in data
- Add peak center time and use it for drift computation (#51)
- Pass record_length as option to DAQReader (#55)
- Make n_top_pmts as option (#34)
- Fix units in plot_energy_spectrum


0.5.0 / 2020-02-05
-------------------
- Natural breaks clustering (#45)
- Save lone hits (#40)
- Store proximity to nearby peaks (#41)
- Add PMT array plot, fixes to mini analysis (#44)
- Bootstrax updates (#47)
- Assume resources are not mutated (do not copy internally)


0.4.1 / 2020-01-18
-------------------
- Fix peak duplication
- Move peak merging code into strax
- Fix documentation build


0.4.0 / 2020-01-17
-------------------
- Peak merging / Two-step clustering (#36)
- Fake DAQ resurrection (#37)
- Matplotlib waveform plotter (#35)
- Updates to get_resource and itp_map from WFsim
- Rename sX_largest_other -> alt_sX_area
- DAQReader fixes (use lz4, time conversion)


0.3.5 / 2019-12-23
------------------
- Integrate peaks with tight_coincidence
- `straxer` script upgrades


0.3.4 / 2019-12-20
-------------------
- Classification tuning (#32)
- Tight coincidence (#32)
- energy spectrum and classification check mini-analyses (#32)
- Bootstrax updates (#33)


0.3.3 / 2019-12-13
------------------
- Fix test data / demo notebook


0.3.2 / 2019-11-13
------------------
- Pulse counting bugfixes (#30)
- Bootstrax:
  - Setup fix (#27)
  - Add correct (epoch-based) run start time (#29)
  - Support compressor config (#29)
- Avoid platform-specific tempfile things (#28)
- Placeholder electron lifetime (#25)


0.3.1 / 2019-09-25
------------------
- Fix resource caching
- Fix tensorflow2 checking (#23)


0.3.0 / 2019-07-21
-------------------
- Mini-analyses, waveform display (#19)
- straxer processing script
- Upgrades to get_resource (#18, #20)
- Require tensorflow2


0.2.2 / 2019-06-17
-------------------
- Upgrade pulse processing and cleanup (#16)


0.2.1 / 2019-06-06
------------------
- Robustness to 0-gain channels, Peaks options available (#15)
- Catch OSError for readonly cache dirs (#14)
- Bootstrax updates (#12, #10)
- Get to_pe and elife from github, add cut plugins (#9)


0.2.0 / 2019-05-04
------------------
- Update records plugin for new pulse processing (#7)
- Move run selection base code into strax (#6)
- Bugfix in s1_min_channels (#5)
- Fix missing export (#4)


0.1.0 / 2018-10-04
------------------
- Split off from the main strax repository
- For earlier history, please see the strax changelogs
