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
- Scada-interface updates (#321,  #324)


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


0.10.0 / 2020-08-187
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
