import straxen
import tempfile
import os
import unittest
import shutil
import uuid

test_run_id_1T = '180423_1021'


class TestBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temp_folder = uuid.uuid4().hex
        # Keep one temp dir because we don't want to download the data every time.
        cls.tempdir = os.path.join(tempfile.gettempdir(), temp_folder)
        assert not os.path.exists(cls.tempdir)

        print("Downloading test data (if needed)")
        st = straxen.contexts.demo()
        cls.run_id = test_run_id_1T
        cls.st = st

    @classmethod
    def tearDownClass(cls):
        # Make sure to only cleanup this dir after we have done all the tests
        if os.path.exists(cls.tempdir):
            shutil.rmtree(cls.tempdir)

    def test_run_selection(self):
        st = self.st
        # Ignore strax-internal warnings
        st.set_context_config({'free_options': tuple(st.config.keys())})

        run_df = st.select_runs(available='raw_records')
        print(run_df)
        run_id = run_df.iloc[0]['name']
        assert run_id == test_run_id_1T

    def test_mini_analysis(self):
        @straxen.mini_analysis(requires=('raw_records',))
        def count_rr(raw_records):
            return len(raw_records)

        n = self.st.count_rr(self.run_id)
        assert n > 100

    @staticmethod
    def _extract_latest_comment(context,
                                test_for_target='raw_records',
                                **context_kwargs,
                                ):
        if context == 'xenonnt_online' and not straxen.utilix_is_configured():
            return
        st = getattr(straxen.contexts, context)(**context_kwargs)
        assert hasattr(st, 'extract_latest_comment'), "extract_latest_comment not added to context?"
        st.extract_latest_comment()
        assert st.runs is not None, "No registry build?"
        assert 'comments' in st.runs.keys()
        st.select_runs(available=test_for_target)
        if context == 'demo':
            assert len(st.runs)
        assert f'{test_for_target}_available' in st.runs.keys()

    def test_extract_latest_comment_nt(self, **opt):
        """Run the test for nt (but only 2000 runs"""
        self._extract_latest_comment(context='xenonnt_online',
                                     minimum_run_number=10_000,
                                     maximum_run_number=12_000,
                                     **opt)

    def test_extract_latest_comment_demo(self):
        self._extract_latest_comment(context='demo')

    def test_extract_latest_comment_lone_hits(self):
        """Run the test for some target that is not in the default availability check"""
        self.test_extract_latest_comment_nt(test_for_target='lone_hits')
