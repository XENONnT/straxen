from immutabledict import immutabledict
import strax
import straxen

common_opts = dict(
    register_all=[
        straxen.pulse_processing,
        straxen.peaklet_processing,
        straxen.peak_processing,
        straxen.event_processing,
        straxen.double_scatter],
    check_available=('raw_records', 'peak_basics'),
    store_run_fields=(
        'name', 'number', 'tags.name',
        'start', 'end', 'livetime', 'mode'))


xnt_common_config = dict(
    n_tpc_pmts=straxen.n_tpc_pmts,
    n_top_pmts=straxen.n_top_pmts,
    gain_model=('to_pe_constant', '1300V_20200428'),
    channel_map=immutabledict(
         # (Minimum channel, maximum channel)
         tpc=(0, 493),
         he=(500, 752),  # high energy
         aqmon=(790, 807),
         tpc_blank=(999, 999),
         mv=(1000, 1083),
         mv_blank=(1999, 1999)))


##
# XENONnT
##

def xenonnt_online(output_folder='./strax_data',
                   we_are_the_daq=False,
                   **kwargs):
    """XENONnT online processing and analysis"""
    context_options = {
        **straxen.contexts.common_opts,
        **kwargs}

    st = strax.Context(
        storage=[
            straxen.RunDB(
                readonly=not we_are_the_daq,
                runid_field='number',
                new_data_path=output_folder),
        ],
        config=straxen.contexts.xnt_common_config,
        **context_options)
    st.register([straxen.DAQReader, straxen.LEDCalibration])

    if not we_are_the_daq:
        st.storage += [
            strax.DataDirectory(
                '/dali/lgrandi/xenonnt/raw',
                readonly=True,
                take_only=straxen.DAQReader.provides),
            strax.DataDirectory(
                '/dali/lgrandi/xenonnt/processed',
                readonly=True)]
        if output_folder:
            st.storage.append(
                strax.DataDirectory(output_folder))

        st.context_config['forbid_creation_of'] = ('raw_records', 'records')

    return st


def xenonnt_led(**kwargs):
    st = xenonnt_online(**kwargs)
    st.context_config['check_available'] = ('raw_records', 'led_calibration')
    # Return a new context with only raw_records and led_calibration registered
    return st.new_context(
        replace=True,
        register=[straxen.DAQReader, straxen.LEDCalibration],
        config=st.config,
        storage=st.storage,
        **st.context_config)


# This gain model is a temp solution untill we have a nice stable one
def xenonnt_simulation(output_folder='./strax_data'):
    import wfsim
    xnt_common_config['gain_model'] = ('to_pe_per_run',
                                        straxen.aux_repo+'58e615f99a4a6b15e97b12951c510de91ce06045/fax_files/to_pe_nt.npy')
    return strax.Context(
        storage=strax.DataDirectory(output_folder),
        register=wfsim.RawRecordsFromFaxNT,
        config=dict(detector='XENONnT',
                    fax_config=straxen.aux_repo+'4e71b8a2446af772c83a8600adc77c0c3b7e54d1/fax_files/fax_config_nt.json',
                    **straxen.contexts.xnt_common_config,
                     ),
        **straxen.contexts.common_opts)


##
# XENON1T
##

x1t_context_config = {
    **common_opts,
    **dict(
        check_available=('raw_records', 'records', 'peaklets',
                             'events', 'event_info'),
        free_options=('channel_map',),
        store_run_fields=tuple(
            [x for x in common_opts['store_run_fields'] if x != 'mode']
            + ['trigger.events_built', 'reader.ini.name']))}

x1t_common_config = dict(
    check_raw_record_overlaps=False,
    allow_sloppy_chunking=True,
    n_tpc_pmts=248,
    n_top_pmts=127,
    channel_map=immutabledict(
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
    # Smaller right extension since we applied the filter
    peak_right_extension=30,
    peak_min_pmts=2,
    save_outside_hits=(3, 3),
    hit_min_amplitude='XENON1T_SR1',
    peak_split_gof_threshold=(
        None,  # Reserved
        ((0.5, 1), (3.5, 0.25)),
        ((2, 1), (4.5, 0.4))))


def demo():
    """Return strax context used in the straxen demo notebook"""
    straxen.download_test_data()
    return strax.Context(
            storage=[strax.DataDirectory('./strax_data'),
                     strax.DataDirectory('./strax_test_data',
                                         deep_scan=True,
                                         provide_run_metadata=True,
                                         readonly=True)],
            register=straxen.RecordsFromPax,
            forbid_creation_of=('raw_records',),
            config=dict(**x1t_common_config),
            **x1t_context_config)


def fake_daq():
    """Context for processing fake DAQ data in the current directory"""
    return strax.Context(
        storage=[strax.DataDirectory('./strax_data'),
                 # Fake DAQ puts run doc JSON in same folder:
                 strax.DataDirectory('./from_fake_daq',
                                     provide_run_metadata=True,
                                     readonly=True)],
        config=dict(daq_input_dir='./from_fake_daq',
                    daq_chunk_duration=int(2e9),
                    daq_compressor='lz4',
                    n_readout_threads=8,
                    daq_overlap_chunk_duration=int(2e8),
                    **x1t_common_config),
        register=straxen.Fake1TDAQReader,
        **x1t_context_config)


def xenon1t_dali(output_folder='./strax_data', build_lowlevel=False):
    return strax.Context(
        storage=[
            strax.DataDirectory(
                '/dali/lgrandi/xenon1t/strax_converted/raw',
                take_only='raw_records',
                provide_run_metadata=True,
                readonly=True),
            strax.DataDirectory(
                '/dali/lgrandi/xenon1t/strax_converted/processed',
                readonly=True),
            strax.DataDirectory(output_folder)],
        register=straxen.RecordsFromPax,
        config=dict(**x1t_common_config),
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=(
            ('raw_records',) if build_lowlevel
            else ('raw_records', 'records', 'peaklets')),
        **x1t_context_config)


def xenon1t_led(**kwargs):
    st = xenon1t_dali(**kwargs)
    st.context_config['check_available'] = ('raw_records', 'led_calibration')
    # Return a new context with only raw_records and led_calibration registered
    return st.new_context(
        replace=True,
        register=[straxen.RecordsFromPax, straxen.LEDCalibration],
        config=st.config,
        storage=st.storage,
        **st.context_config)


def xenon1t_simulation(output_folder='./strax_data'):
    import wfsim
    return strax.Context(
        storage=strax.DataDirectory(output_folder),
        register=wfsim.RawRecordsFromFax1T,
        config=dict(fax_config=straxen.aux_repo+'1c5793b7d6c1fdb7f99a67926ee3c16dd3aa944f/fax_files/fax_config_1t.json',
                    detector='XENON1T',
                    **straxen.contexts.x1t_common_config),
        **straxen.contexts.common_opts)
