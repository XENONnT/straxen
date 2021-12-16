import strax
import straxen
import tarfile
import io
import os
from warnings import warn
from os import environ as os_environ
from straxen import aux_repo, pax_file
from pandas import DataFrame
from immutabledict import immutabledict
import numpy as np


export, __all__ = strax.exporter()


nt_test_run_id = '012882'
test_run_id_1T = '180423_1021'

testing_config_1T = dict(
    hev_gain_model=('1T_to_pe_placeholder', False),
    gain_model=('1T_to_pe_placeholder', False),
    elife=1e6,
    electron_drift_velocity=("electron_drift_velocity_constant", 1e-4),
    electron_drift_time_gate=("electron_drift_time_gate_constant", 1700),
)

# Let's make a dummy map for NVeto
_nveto_pmt_dummy = {'channel': list(range(2000, 2120)),
                    'x': list(range(120)),
                    'y': list(range(120)),
                    'z': list(range(120)),
                    }
_nveto_pmt_dummy_df = DataFrame(_nveto_pmt_dummy)

# Some configs are better obtained from the strax_auxiliary_files repo.
# Let's use small files, we don't want to spend a lot of time downloading
# some file.
_testing_config_nT = dict(
    nn_architecture=
    aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.json',
    nn_weights=
    aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.h5',
    gain_model=("to_pe_placeholder", True),
    elife=1e6,
    baseline_samples_nv=10,
    fdc_map=pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz'),
    gain_model_nv=("adc_nv", True),
    gain_model_mv=("adc_mv", True),
    nveto_pmt_position_map=_nveto_pmt_dummy,
    s1_xyz_map=f'itp_map://resource://{pax_file("XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json")}?fmt=json',
    s2_xy_map=f'itp_map://resource://{pax_file("XENON1T_s2_xy_ly_SR1_v2.2.json")}?fmt=json',
    electron_drift_velocity=("electron_drift_velocity_constant", 1e-4),
    s1_aft_map=aux_repo + '023cb8caf2008b289664b0fefc36b1cebb45bbe4/strax_files/s1_aft_UNITY_xyz_XENONnT.json',  # noqa
    s2_optical_map=aux_repo + '9891ee7a52fa00e541480c45ab7a1c9a72fcffcc/strax_files/XENONnT_s2_xy_unity_patterns.json.gz',  # noqa
    s1_optical_map=aux_repo + '9891ee7a52fa00e541480c45ab7a1c9a72fcffcc/strax_files/XENONnT_s1_xyz_unity_patterns.json.gz',  # noqa
    electron_drift_time_gate=("electron_drift_time_gate_constant", 2700),
    hit_min_amplitude='pmt_commissioning_initial',
    hit_min_amplitude_nv=20,
    hit_min_amplitude_mv=80,
    hit_min_amplitude_he='pmt_commissioning_initial_he',
    avg_se_gain=1.0,
    se_gain=1.0,
    rel_extraction_eff=1.0,
    g1=0.1,
    g2=10
)


@export
def download_test_data(test_data='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/353b2c60a01e96f67e4ba544ce284bd91241964d/strax_files/strax_test_data_straxv1.1.0.tar',  #  noqa
                       ):
    """Downloads strax test data to strax_test_data in the current directory"""
    blob = straxen.common.get_resource(test_data, fmt='binary')
    f = io.BytesIO(blob)
    tf = tarfile.open(fileobj=f)
    tf.extractall()


@export
def _overwrite_testing_function_file(function_file):
    """For testing purposes allow this function file to be loaded from HOME/testing_folder"""
    if not _is_on_pytest():
        # If we are not on a pytest, never try using a local file.
        return function_file

    home = os.environ.get('HOME')
    if home is None:
        # Impossible to load from non-existent folder
        return function_file

    testing_file = os.path.join(home, function_file)

    if os.path.exists(testing_file):
        # For testing purposes allow loading from 'home/testing_folder'
        warn(f'Using local function: {function_file} from {testing_file}! '
             f'If you are not integrated testing on github you should '
             f'absolutely remove this file. (See #559)')
        function_file = testing_file

    return function_file


@export
def _is_on_pytest():
    """Check if we are on a pytest"""
    return 'PYTEST_CURRENT_TEST' in os_environ


def nt_test_context(target_context='xenonnt_online',
                    deregister = ('peak_veto_tags', 'events_tagged'),
                    **kwargs):
    st = getattr(straxen.contexts, target_context)(**kwargs)
    st._plugin_class_registry['raw_records'].__version__ = "MOCKTESTDATA"  # noqa
    st.storage = [strax.DataDirectory('./strax_test_data')]
    download_test_data('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/1d3706d4b47cbd23b5cae66d5e258bb84487ad01/strax_files/012882-raw_records-z7q2d2ye2t.tar')  # noqa
    assert st.is_stored(nt_test_run_id, 'raw_records'), os.listdir(st.storage[-1].path)

    to_remove = list(deregister)
    if not straxen.utilix_is_configured(warning_message=False):
        st.set_config(_testing_config_nT)
        to_remove += 'peak_positions_mlp peak_positions_cnn peak_positions_gcn s2_recon_pos_diff'.split()  # noqa
        # TODO The test data for this plugin doesn't work
        to_remove += ['event_pattern_fit']
        st.set_config({'gain_model': ("to_pe_placeholder", True)})
        st.register(straxen.PeakPositions1T)
    for plugin in to_remove:
        del st._plugin_class_registry[plugin]
    return st


@strax.takes_config(
    strax.Option('secret_time_offset', default=0, track=False),
    strax.Option('recs_per_chunk', default=10, track=False),
    strax.Option('n_chunks', default=2, track=False,
                 help='Number of chunks for the dummy raw records we are writing here'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="frozendict mapping subdetector to (min, max) "
                      "channel number.")
)
class DummyRawRecords(strax.Plugin):
    """
    Provide dummy raw records for the mayor raw_record types
    """
    provides = ('raw_records',
                'raw_records_he',
                'raw_records_nv',
                'raw_records_aqmon',
                'raw_records_aux_mv',
                'raw_records_mv'
                )
    parallel = 'process'
    depends_on = tuple()
    data_kind = immutabledict(zip(provides, provides))
    rechunk_on_save = False
    dtype = {p: strax.raw_record_dtype() for p in provides}

    def setup(self):
        self.channel_map_keys = {'he': 'he',
                                 'nv': 'nveto',
                                 'aqmon': 'aqmon',
                                 'aux_mv': 'aux_mv',
                                 's_mv': 'mv',
                                 }  # s_mv otherwise same as aux in endswith

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.config['n_chunks']

    def compute(self, chunk_i):
        t0 = chunk_i + self.config['secret_time_offset']
        if chunk_i < self.config['n_chunks'] - 1:
            # One filled chunk
            r = np.zeros(self.config['recs_per_chunk'], self.dtype['raw_records'])
            r['time'] = t0
            r['length'] = r['dt'] = 1
            r['channel'] = np.arange(len(r))
        else:
            # One empty chunk
            r = np.zeros(0, self.dtype['raw_records'])

        res = {}
        for p in self.provides:
            rr = np.copy(r)
            # Add detector specific channel offset:
            for key, channel_key in self.channel_map_keys.items():
                if channel_key not in self.config['channel_map']:
                    # Channel map for 1T is different.
                    continue
                if p.endswith(key):
                    s, e = self.config['channel_map'][channel_key]
                    rr['channel'] += s
            res[p] = self.chunk(start=t0, end=t0 + 1, data=rr, data_type=p)
        return res
