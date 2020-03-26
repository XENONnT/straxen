import warnings

from frozendict import frozendict
import strax
import straxen


common_opts = dict(
   register_all=[
       straxen.nveto_daqreader,
       straxen.pulse_processing,
       straxen.peaklet_processing,
       straxen.peak_processing,
       straxen.event_processing],
   store_run_fields=(
       'name', 'number',
       'reader.ini.name', 'tags.name',
       'start', 'end', 'livetime',
       'trigger.events_built'),
   check_available=('raw_records', 'nveto_pre_raw_records', 'records', 'peaklets',
                    'events', 'event_info'))

x1t_common_config = dict(
    check_raw_record_overlaps=False,
    n_tpc_pmts=248,
    channel_map=frozendict(
        # (Minimum channel, maximum channel)
        tpc=(0, 247),
        diagnostic=(248, 253),
        aqmon=(254, 999)),
    hev_gain_model=('to_pe_per_run',
                    'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy'),
    gain_model=('to_pe_per_run',
                'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy'),
    pmt_pulse_filter=(
        0.012, -0.119,
        2.435, -1.271, 0.357, -0.174, -0., -0.036,
        -0.028, -0.019, -0.025, -0.013, -0.03, -0.039,
        -0.005, -0.019, -0.012, -0.015, -0.029, 0.024,
        -0.007, 0.007, -0.001, 0.005, -0.002, 0.004, -0.002),
    tail_veto_threshold=int(1e5),
    save_outside_hits=(3, 3),
    hit_min_amplitude=straxen.adc_thresholds(),
    # Some setting which are required by the fake daq for the nVETO HdM test set-up but not needed otherwise:
    n_nveto_pmts=32,
)

xnt_common_config = dict(
    n_tpc_pmts=493,
    n_nveto_pmts=120,
    gain_model=('to_pe_constant',
                0.005),
    channel_map=frozendict(
         # (Minimum channel, maximum channel)
         tpc=(0, 493),
         he=(500, 752),  # high energy
         aqmon=(799, 807),
         tpc_blank=(999, 999),
         mv=(1000, 1083),
         mv_blank=(1999, 1999),
         nveto=(2000, 2119),
         nveto_aqmon=(808, 815),
         nveto_blank=(2999),
    )
)



##
# XENON1T
##

def fake_daq():
  """Context for processing fake DAQ data in the current directory"""
  return strax.Context(
      storage=[strax.DataDirectory('./strax_data',
                                   provide_run_metadata=False),
               # Fake DAQ puts run doc JSON in same folder:
               strax.DataDirectory('./from_fake_daq',
                                   readonly=True)],
      config=dict(daq_input_dir='./from_fake_daq',
                  daq_chunk_duration=int(2e9),
                  daq_compressor='lz4',
                 n_readout_threads=8,
                  daq_overlap_chunk_duration=int(2e8),
                 **x1t_common_config),
      register=straxen.Fake1TDAQReader,
      **common_opts)


nveto_common_opts = dict(
    register_all=[
       #  straxen.nveto_daqreader,
       # straxen.nveto_recorder,
       # straxen.nveto_pulse_processing
    ],
    store_run_fields=(
        'name', 'number',
        'reader.ini.name', 'tags.name',
        'start', 'end', 'livetime',
        'trigger.events_built'),
    check_available=('nveto_pre_raw_records',
                    # 'nveto_raw_records',
                    # 'nveto_diagnostic_lone_records',
                    # 'nveto_lone_records_count',
                    # 'nveto_records',
                    # 'nveto_pulses',
                    # 'nveto_pulse_basics',
                     ))

def strax_nveto_hdm_test():
    return strax.Context(
        storage=[
            strax.DataDirectory(
                # '/dali/lgrandi/hiraide/data/nveto/DaqTestHdM',
                '/dali/lgrandi/wenz/strax_data/HdMdata_strax_v0_9_0/strax_data_raw',
                take_only='nveto_pre_raw_records',
                deep_scan=False,
                readonly=True),
            strax.DataDirectory(
                '/dali/lgrandi/wenz/strax_data/HdMdata_strax_v0_9_0/strax_data',
                provide_run_metadata=False)],
        config=HdM_common_config,
        **nveto_common_opts)
