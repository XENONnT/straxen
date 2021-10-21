import strax
import straxen
import tarfile
import io
import os
from warnings import warn
from os import environ as os_environ
from straxen import aux_repo, pax_file
nt_test_run_id = '012882'
export, __all__ = strax.exporter()


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
                    **kwargs):
    st = getattr(straxen.contexts, target_context)(**kwargs)
    st._plugin_class_registry['raw_records'].__version__ = "MOCKTESTDATA"
    st.storage = [strax.DataDirectory('./strax_test_data')]
    download_test_data('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/8b304bde43260eb47b4d666c244a386ac5a25b51/strax_files/012882-raw_records-z7q2d2ye2t.tar')
    assert st.is_stored(nt_test_run_id, 'raw_records'), os.listdir(st.storage[-1].path)

    if not straxen.utilix_is_configured(warning_message=False):
        st.set_config(_testing_config_nT)
        del st._plugin_class_registry['peak_positions_mlp']
        del st._plugin_class_registry['peak_positions_cnn']
        del st._plugin_class_registry['peak_positions_gcn']
        del st._plugin_class_registry['s2_recon_pos_diff']
        st.register(straxen.PeakPositions1T)
        st.set_config({'gain_model': ("to_pe_placeholder", True)})
        print(f"Using {st._plugin_class_registry['peak_positions']} for posrec tests")
    return st


# Let's make a dummy map for NVeto
_nveto_pmt_dummy_df = {'channel': list(range(2000, 2120)),
                      'x': list(range(120)),
                      'y': list(range(120)),
                      'z': list(range(120))}

# Some configs are better obtained from the strax_auxiliary_files repo.
# Let's use small files, we don't want to spend a lot of time downloading
# some file.
_testing_config_nT = dict(
    nn_architecture=
    aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.json',
    nn_weights=
    aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.h5',
    gain_model=("to_pe_placeholder", True),
    s2_xy_correction_map=pax_file('XENON1T_s2_xy_ly_SR0_24Feb2017.json'),
    elife_conf=('elife_constant', 1e6),
    baseline_samples_nv=10,
    fdc_map=pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz'),
    gain_model_nv=("adc_nv", True),
    gain_model_mv=("adc_mv", True),
    nveto_pmt_position_map=_nveto_pmt_dummy_df,
    s1_xyz_correction_map=pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR0_pax-680_fdc-3d_v0.json'),
    electron_drift_velocity=("electron_drift_velocity_constant", 1e-4),
    s1_aft_map=aux_repo + 'ffdadba3439ae7922b19f5dd6479348b253c09b0/strax_files/s1_aft_UNITY_xyz_XENONnT.json',
    s2_optical_map=aux_repo + '8a6f0c1a4da4f50546918cd15604f505d971a724/strax_files/s2_map_UNITY_xy_XENONnT.json',
    s1_optical_map=aux_repo + '8a6f0c1a4da4f50546918cd15604f505d971a724/strax_files/s1_lce_UNITY_xyz_XENONnT.json',
    electron_drift_time_gate=("electron_drift_time_gate_constant", 2700),
    hit_min_amplitude='pmt_commissioning_initial',
    hit_min_amplitude_nv=20,
    hit_min_amplitude_mv=80,
    hit_min_amplitude_he='pmt_commissioning_initial_he'
)
