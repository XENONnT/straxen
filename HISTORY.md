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
