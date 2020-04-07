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
