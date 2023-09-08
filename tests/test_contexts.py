"""For all of the context, do a quick check to see that we are able to search
a field (i.e. can build the dependencies in the context correctly)
See issue #233 and PR #236"""
from straxen.contexts import xenon1t_dali, xenon1t_led, fake_daq, demo
from straxen.contexts import xenonnt_led, xenonnt_online, xenonnt
import straxen
import tempfile
import os
import unittest


##
# XENONnT
##


def test_xenonnt_online():
    st = xenonnt_online(_database_init=False)
    st.search_field('time')


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xenonnt_online_with_online_frontend():
    st = xenonnt_online(include_online_monitor=True)
    for sf in st.storage:
        if 'OnlineMonitor' == sf.__class__.__name__:
            break
    else:
        raise ValueError(f"Online monitor not in {st.storage}")


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xenonnt_online_rucio_local():
    st = xenonnt_online(include_rucio_local=True, _rucio_local_path='./test')
    for sf in st.storage:
        if 'RucioLocalFrontend' == sf.__class__.__name__:
            break
    else:
        raise ValueError(f"Online monitor not in {st.storage}")


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_xennonnt():
    st = xenonnt(_database_init=False)
    st.search_field('time')


def test_xenonnt_led():
    st = xenonnt_led(_database_init=False)
    st.search_field('time')


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_nt_is_nt_online():
    # Test that nT and nT online are the same
    st_online = xenonnt_online(_database_init=False)

    st = xenonnt(_database_init=False)
    for plugin in st._plugin_class_registry.keys():
        print(f'Checking {plugin}')
        nt_key = st.key_for('0', plugin)
        nt_online_key = st_online.key_for('0', plugin)
        assert str(nt_key) == str(nt_online_key)


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_offline():
    """
    Let's try and see which CMT versions are compatible with this straxen
    version
    """
    cmt = straxen.CorrectionsManagementServices()
    cmt_versions = list(cmt.global_versions)[::-1]
    print(cmt_versions)
    success_for = []
    for global_version in cmt_versions:
        try:
            xenonnt(global_version)
            success_for.append(global_version)
        except straxen.CMTVersionError:
            pass
    print(f'This straxen version works with {success_for} but is '
          f'incompatible with {set(cmt_versions)-set(success_for)}')

    test = unittest.TestCase()
    # We should always work for one offline and the online version
    test.assertTrue(len(success_for) >= 2)


##
# XENON1T
##


def test_xenon1t_dali():
    st = xenon1t_dali()
    st.search_field('time')


def test_demo():
    """
    Test the demo context. Since we download the folder to the current
        working directory, make sure we are in a tempfolder where we
        can write the data to
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Temporary directory is ", temp_dir)
            os.chdir(temp_dir)
            st = demo()
            st.search_field('time')
        # On windows, you cannot delete the current process'
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')


def test_fake_daq():
    st = fake_daq()
    st.search_field('time')


def test_xenon1t_led():
    st = xenon1t_led()
    st.search_field('time')

##
# WFSim
##


# Simulation contexts are only tested when special flags are set

@unittest.skipIf('ALLOW_WFSIM_TEST' not in os.environ,
                 "if you want test wfsim context do `export 'ALLOW_WFSIM_TEST'=1`")
class TestSimContextNT(unittest.TestCase):
    @staticmethod
    def context(*args, **kwargs):
        kwargs.setdefault('cmt_version', 'global_ONLINE')
        return straxen.contexts.xenonnt_simulation(*args, **kwargs)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_sim_context_main(self):
        st = self.context(cmt_run_id_sim='008000')
        st.search_field('time')

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_sim_context_alt(self):
        """Some examples of how to run with a custom WFSim context"""
        self.context(cmt_run_id_sim='008000', cmt_run_id_proc='008001')
        self.context(cmt_run_id_sim='008000',
                     cmt_option_overwrite_sim={'elife': 1e6})

        self.context(cmt_run_id_sim='008000',
                     overwrite_fax_file_sim={'elife': 1e6})

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_diverging_context_options(self):
        """
        Test diverging options. Idea is that you can use different
        settings for processing and generating data, should have been
        handled by RawRecordsFromWFsim but is now hacked into the
        xenonnt_simulation context

        Just to show how convoluted this syntax for the
        xenonnt_simulation context / CMT is...
        """
        self.context(cmt_run_id_sim='008000',
                     cmt_option_overwrite_sim={'elife': ('elife_constant', 1e6, True)},
                     cmt_option_overwrite_proc={'elife': ('elife_constant', 1e5, True)},
                     overwrite_from_fax_file_proc=True,
                     overwrite_from_fax_file_sim=True,
                     _config_overlap={'electron_lifetime_liquid': 'elife'},
                     )

    def test_nt_sim_context_bad_inits(self):
        with self.assertRaises(RuntimeError):
            self.context(cmt_run_id_sim=None, cmt_run_id_proc=None,)

@unittest.skipIf('ALLOW_WFSIM_TEST' not in os.environ,
                 "if you want test wfsim context do `export 'ALLOW_WFSIM_TEST'=1`")
def test_sim_context():
    st = straxen.contexts.xenon1t_simulation()

@unittest.skipIf(
    "ALLOW_WFSIM_TEST" not in os.environ,
    "if you want test wfsim context do `export 'ALLOW_WFSIM_TEST'=1`",
)
def test_sim_offline_context():
    st = straxen.contexts.xenonnt_simulation_offline(
        run_id="026000",
        global_version="global_v11",
        fax_config="fax_config_nt_sr0_v4.json",
    )


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_offline():
    st = xenonnt('latest')
    st.provided_dtypes()
