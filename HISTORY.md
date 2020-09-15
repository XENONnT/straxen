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
