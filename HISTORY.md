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
