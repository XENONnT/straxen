from immutabledict import immutabledict
import strax
import straxen
from straxen.common import pax_file


def get_x1t_context_config():
    """Wrapper to get straxen.contexts after imports."""
    from straxen.contexts import common_opts

    x1t_context_config = {
        **common_opts,
        **dict(
            check_available=("raw_records", "records", "peaklets", "events", "event_info"),
            free_options=("channel_map",),
            use_per_run_defaults=True,
            store_run_fields=tuple(
                [x for x in common_opts["store_run_fields"] if x != "mode"]
                + ["trigger.events_built", "reader.ini.name"]
            ),
        ),
    }
    x1t_context_config.update(
        dict(
            register=common_opts["register"]
            + [
                straxen.PeakPositions1T,
                straxen.RecordsFromPax,
                straxen.EventInfo1T,
            ]
        )
    )
    return x1t_context_config


x1t_common_config = dict(
    check_raw_record_overlaps=False,
    allow_sloppy_chunking=True,
    n_tpc_pmts=248,
    n_top_pmts=127,
    channel_map=immutabledict(
        # (Minimum channel, maximum channel)
        tpc=(0, 247),
        diagnostic=(248, 253),
        aqmon=(254, 999),
    ),
    # Records
    hev_gain_model="cmt://to_pe_model?version=v1&detector=1t&run_id=plugin.run_id",
    pmt_pulse_filter=(
        0.012,
        -0.119,
        2.435,
        -1.271,
        0.357,
        -0.174,
        -0.0,
        -0.036,
        -0.028,
        -0.019,
        -0.025,
        -0.013,
        -0.03,
        -0.039,
        -0.005,
        -0.019,
        -0.012,
        -0.015,
        -0.029,
        0.024,
        -0.007,
        0.007,
        -0.001,
        0.005,
        -0.002,
        0.004,
        -0.002,
    ),
    hit_min_amplitude="legacy-thresholds://XENON1T_SR1",
    tail_veto_threshold=int(1e5),
    save_outside_hits=(3, 3),
    # Peaklets
    peaklet_gap_threshold=350,
    gain_model="cmt://to_pe_model?version=v1&detector=1t&run_id=plugin.run_id",
    peak_split_gof_threshold=(None, ((0.5, 1), (3.5, 0.25)), ((2, 1), (4.5, 0.4))),  # Reserved
    peak_min_pmts=2,
    # MergedS2s
    s2_merge_gap_thresholds=((1.7, 5.0e3), (4.0, 500.0), (5.0, 0.0)),
    # Peaks
    # Smaller right extension since we applied the filter
    peak_right_extension=30,
    s1_max_rise_time_post100=150,
    s1_min_coincidence=3,
    event_s1_min_coincidence=3,
    # Events*
    left_event_extension=int(0.3e6),
    right_event_extension=int(1e6),
    elife=1e6,
    electron_drift_velocity=1.3325e-4,
    max_drift_length=96.9,
    electron_drift_time_gate=1700,
    se_gain=28.2,
    avg_se_gain=28.2,
    rel_extraction_eff=1.0,
    rel_light_yield=1.0,
    s1_xyz_map=f'itp_map://resource://{pax_file("XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json")}?fmt=json',  # noqa
    s2_xy_map=f'itp_map://resource://{pax_file("XENON1T_s2_xy_ly_SR1_v2.2.json")}?fmt=json',
    g1=0.1426,
    g2=11.55 / (1 - 0.63),
)


def demo():
    """Return strax context used in the straxen demo notebook."""
    straxen.download_test_data()

    st = strax.Context(
        storage=[
            strax.DataDirectory("./strax_data"),
            strax.DataDirectory(
                "./strax_test_data", deep_scan=True, provide_run_metadata=True, readonly=True
            ),
        ],
        forbid_creation_of=straxen.daqreader.DAQReader.provides,
        config=dict(**x1t_common_config),
        **get_x1t_context_config(),
    )

    # Use configs that are always available
    st.set_config(
        dict(
            hev_gain_model="legacy-to-pe://1T_to_pe_placeholder",
            gain_model="legacy-to-pe://1T_to_pe_placeholder",
            elife=1e6,
            electron_drift_velocity=1.3325e-4,
            se_gain=28.2,
            avg_se_gain=28.2,
            rel_extraction_eff=1.0,
            s1_xyz_map=f'itp_map://resource://{pax_file("XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json")}?fmt=json',
            s2_xy_map=f'itp_map://resource://{pax_file("XENON1T_s2_xy_ly_SR1_v2.2.json")}?fmt=json',
        )
    )
    return st


def fake_daq():
    """Context for processing fake DAQ data in the current directory."""
    st = strax.Context(
        storage=[
            strax.DataDirectory("./strax_data"),
            # Fake DAQ puts run doc JSON in same folder:
            strax.DataDirectory("./from_fake_daq", provide_run_metadata=True, readonly=True),
        ],
        config=dict(
            daq_input_dir="./from_fake_daq",
            daq_chunk_duration=int(2e9),
            daq_compressor="lz4",
            n_readout_threads=8,
            daq_overlap_chunk_duration=int(2e8),
            **x1t_common_config,
        ),
        **get_x1t_context_config(),
    )
    st.register(straxen.Fake1TDAQReader)
    return st


def xenon1t_dali(output_folder="./strax_data", build_lowlevel=False, **kwargs):
    context_options = {**get_x1t_context_config(), **kwargs}

    st = strax.Context(
        storage=[
            strax.DataDirectory(
                "/dali/lgrandi/xenon1t/strax_converted/raw",
                take_only="raw_records",
                provide_run_metadata=True,
                readonly=True,
            ),
            strax.DataDirectory("/dali/lgrandi/xenon1t/strax_converted/processed", readonly=True),
            strax.DataDirectory(output_folder),
        ],
        config=dict(**x1t_common_config),
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=(
            straxen.daqreader.DAQReader.provides
            if build_lowlevel
            else straxen.daqreader.DAQReader.provides + ("records", "peaklets")
        ),
        **context_options,
    )
    return st


def xenon1t_led(**kwargs):
    st = xenon1t_dali(**kwargs)
    st.set_context_config(
        {
            "check_available": ("raw_records", "led_calibration"),
            "free_options": list(get_x1t_context_config().keys()),
        }
    )
    # Return a new context with only raw_records and led_calibration registered
    st = st.new_context(replace=True, config=st.config, storage=st.storage, **st.context_config)
    st.register([straxen.RecordsFromPax, straxen.LEDCalibration])
    return st


def xenon1t_simulation(output_folder="./strax_data"):
    import wfsim

    st = strax.Context(
        storage=strax.DataDirectory(output_folder),
        config=dict(fax_config="fax_config_1t.json", detector="XENON1T", **x1t_common_config),
        **get_x1t_context_config(),
    )
    st.register(wfsim.RawRecordsFromFax1T)
    st.deregister_plugins_with_missing_dependencies()
    return st
