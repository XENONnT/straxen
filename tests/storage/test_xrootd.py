import json
import os
import unittest
import numpy as np
import strax
import straxen

try:
    import fsspec

    HAVE_FSSPEC = True
except ImportError:
    HAVE_FSSPEC = False


class DummyPlugin(strax.Plugin):
    provides = "test_data"
    depends_on: tuple = tuple()
    data_kind = "test_data"
    dtype = strax.time_fields


@unittest.skipIf(not HAVE_FSSPEC, "fsspec is not installed")
class TestXRootD(unittest.TestCase):
    """Test XRootDBackend and XRootDFrontend streaming storage functionality."""

    def setUp(self):
        self.fs = fsspec.filesystem("memory")
        self.run_id = "050000"
        self.data_type = "test_data"
        self.subpath = "/processed"
        self.redirector = "memory://"

        self.dtype = strax.time_fields
        self.data = np.zeros(10, dtype=self.dtype)
        self.data["time"] = np.arange(0, 100, 10)
        self.data["endtime"] = self.data["time"] + 10

        self.key = strax.DataKey(
            run_id=self.run_id,
            data_type=self.data_type,
            lineage={self.data_type: ["DummyPlugin", "0.0.0", {}]},
        )
        self.folder_name = str(self.key)
        self.full_folder = f"{self.subpath}/{self.folder_name}"
        self.fs.mkdir(self.full_folder)

    def tearDown(self):
        try:
            if self.fs.exists(self.subpath):
                self.fs.rm(self.subpath, recursive=True)
        except Exception:
            pass

    def _write_chunk_and_metadata(self, compressor="zstd", metadata_type="modern"):
        target_folder = f"{self.full_folder}_temp" if metadata_type == "temp" else self.full_folder
        self.fs.makedirs(target_folder, exist_ok=True)
        chunk_fn = f"{self.data_type}-{self.key.lineage_hash}-000000"
        chunk_path = f"{target_folder}/{chunk_fn}"

        with self.fs.open(chunk_path, mode="wb") as f:
            strax.save_file(f, self.data, compressor=compressor)

        metadata = {
            "strax_version": strax.__version__,
            "run_id": self.run_id,
            "data_type": self.data_type,
            "data_kind": self.data_type,
            "dtype": self.data.dtype.descr.__repr__(),
            "compressor": compressor,
            "lineage_hash": self.key.lineage_hash,
            "lineage": self.key.lineage,
            "writing_ended": 1,
            "chunks": [
                {
                    "filename": chunk_fn,
                    "n": len(self.data),
                    "start": int(self.data["time"][0]),
                    "end": int(self.data["time"][-1] + 10),
                    "run_id": self.run_id,
                }
            ],
        }

        if metadata_type in ("modern", "temp"):
            prefix = f"{self.data_type}-{self.key.lineage_hash}"
            md_fn = strax.RUN_METADATA_PATTERN % prefix
            md_path = f"{target_folder}/{md_fn}"
        elif metadata_type == "legacy":
            md_path = f"{target_folder}/metadata.json"
        else:
            raise ValueError(f"Unknown metadata_type {metadata_type}")

        with self.fs.open(md_path, mode="wb") as f:
            f.write(json.dumps(metadata).encode("utf-8"))

        return chunk_fn

    def test_url_builder(self):
        frontend = straxen.XRootDFrontend(
            redirector_url="root://midway-origin.uchicago.edu/",
            subpath="xenon/xenonnt/processed",
        )
        url = frontend.build_url(self.key)
        self.assertEqual(
            url,
            f"root://midway-origin.uchicago.edu//xenon/xenonnt/processed/{self.folder_name}",
        )

        # Without trailing slash on redirector
        frontend2 = straxen.XRootDFrontend(
            redirector_url="root://midway-origin.uchicago.edu:1094",
            subpath="xenon/xenonnt/processed/",
        )
        self.assertEqual(
            frontend2.build_url(self.key),
            f"root://midway-origin.uchicago.edu:1094//xenon/xenonnt/processed/{self.folder_name}",
        )

        # Without subpath
        frontend3 = straxen.XRootDFrontend(
            redirector_url="root://midway-origin.uchicago.edu/",
            subpath="",
        )
        self.assertEqual(
            frontend3.build_url(self.key),
            f"root://midway-origin.uchicago.edu//{self.folder_name}",
        )

        # Memory / file URL
        frontend_mem = straxen.XRootDFrontend(
            redirector_url="memory://",
            subpath="processed",
        )
        self.assertEqual(
            frontend_mem.build_url(self.key),
            f"memory:///processed/{self.folder_name}",
        )

    def test_metadata_retrieval_modern_and_caching(self):
        self._write_chunk_and_metadata(metadata_type="modern")
        backend = straxen.XRootDBackend(cache_metadata=True)
        url = f"memory://{self.full_folder}"

        # First retrieval fetches and caches
        self.assertEqual(len(backend._metadata_cache), 0)
        md1 = backend.get_metadata(url)
        self.assertEqual(len(backend._metadata_cache), 1)

        # Second retrieval uses cache
        md2 = backend.get_metadata(url)
        self.assertIs(md1, md2)

        # Clear cache
        backend.clear_metadata_cache()
        self.assertEqual(len(backend._metadata_cache), 0)

    def test_metadata_retrieval_legacy(self):
        self._write_chunk_and_metadata(metadata_type="legacy")
        backend = straxen.XRootDBackend()
        url = f"memory://{self.full_folder}"

        md = backend.get_metadata(url)
        self.assertEqual(md["run_id"], self.run_id)

    def test_metadata_retrieval_temp(self):
        self._write_chunk_and_metadata(metadata_type="temp")
        backend = straxen.XRootDBackend()
        url = f"memory://{self.full_folder}_temp"

        md = backend.get_metadata(url)
        self.assertEqual(md["run_id"], self.run_id)

    def test_missing_and_corrupted_metadata(self):
        backend = straxen.XRootDBackend()
        nonexistent_url = "memory:///processed/nonexistent-key-0000"

        with self.assertRaises(strax.DataNotAvailable):
            backend.get_metadata(nonexistent_url)

        # Corrupted JSON
        corrupted_folder = "/processed/050000-corrupted-lineage"
        self.fs.mkdir(corrupted_folder)
        with self.fs.open(f"{corrupted_folder}/metadata.json", "wb") as f:
            f.write(b"this is not valid json")

        with self.assertRaises(strax.DataCorrupted):
            backend.get_metadata(f"memory://{corrupted_folder}")

        # Test frontend find directly propagates DataCorrupted (Bug 7)
        corrupted_key = strax.DataKey(
            run_id="050000",
            data_type="corrupted",
            lineage={"corrupted": ["CorruptedPlugin", "0.0.0", {}]},
        )
        fe_folder = f"/processed/{corrupted_key}"
        self.fs.makedirs(fe_folder, exist_ok=True)
        with self.fs.open(f"{fe_folder}/metadata.json", "wb") as f:
            f.write(b"this is not valid json")

        fe = straxen.XRootDFrontend(
            redirector_url="memory://",
            subpath="processed",
        )
        with self.assertRaises(strax.DataCorrupted):
            fe.find(corrupted_key)

    def test_read_chunk_compressors(self):
        for comp in ["zstd", "blosc", "lz4"]:
            chunk_fn = self._write_chunk_and_metadata(compressor=comp)
            backend = straxen.XRootDBackend()
            url = f"memory://{self.full_folder}"

            chunk_info = {"filename": chunk_fn}
            loaded_data = backend._read_chunk(
                url, chunk_info=chunk_info, dtype=self.dtype, compressor=comp
            )
            np.testing.assert_array_equal(loaded_data, self.data)

    def test_missing_chunk(self):
        self._write_chunk_and_metadata()
        backend = straxen.XRootDBackend()
        url = f"memory://{self.full_folder}"
        chunk_info = {"filename": "missing_chunk_file"}

        with self.assertRaises(strax.DataNotAvailable):
            backend._read_chunk(url, chunk_info=chunk_info, dtype=self.dtype, compressor="zstd")

    def test_saver_not_implemented(self):
        backend = straxen.XRootDBackend()
        with self.assertRaises(NotImplementedError):
            backend.saver("memory:///test", {})
        with self.assertRaises(NotImplementedError):
            backend._saver("memory:///test", {})

    def test_frontend_find_and_several(self):
        self._write_chunk_and_metadata()
        frontend = straxen.XRootDFrontend(
            redirector_url=self.redirector,
            subpath=self.subpath.lstrip("/"),
        )

        backend_name, backend_key = frontend.find(self.key)
        self.assertEqual(backend_name, "XRootDBackend")
        self.assertEqual(backend_key, f"memory://{self.full_folder}")

        # Write attempt must raise DataNotAvailable
        with self.assertRaises(strax.DataNotAvailable):
            frontend.find(self.key, write=True)

        # Missing key
        missing_key = strax.DataKey(
            run_id="999999",
            data_type=self.data_type,
            lineage=self.key.lineage,
        )
        with self.assertRaises(strax.DataNotAvailable):
            frontend.find(missing_key)

        # find_several
        results = frontend.find_several([self.key, missing_key])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "XRootDBackend")
        self.assertFalse(results[1])

    def test_strax_context_streaming(self):
        self._write_chunk_and_metadata(compressor="zstd")
        frontend = straxen.XRootDFrontend(
            redirector_url=self.redirector,
            subpath=self.subpath.lstrip("/"),
        )

        st = strax.Context(storage=[frontend])
        st.register(DummyPlugin)

        self.assertTrue(st.is_stored(self.run_id, self.data_type))
        arr = st.get_array(self.run_id, self.data_type)
        self.assertEqual(len(arr), len(self.data))
        np.testing.assert_array_equal(arr, self.data)

    def test_context_builder_option(self):
        st = straxen.contexts.xenonnt(
            include_xrootd=True,
            _xrootd_url=self.redirector,
            _xrootd_subpath=self.subpath.lstrip("/"),
            _database_init=False,
        )
        xrootd_frontends = [sf for sf in st.storage if isinstance(sf, straxen.XRootDFrontend)]
        self.assertEqual(len(xrootd_frontends), 1)
        self.assertEqual(xrootd_frontends[0].redirector_url, "memory://")

    def test_missing_fsspec_raises(self):
        import unittest.mock

        with unittest.mock.patch("straxen.storage.xrootd.HAVE_FSSPEC", False):
            with self.assertRaises(ImportError):
                straxen.XRootDBackend()
            with self.assertRaises(ImportError):
                straxen.XRootDFrontend()

    def test_dynamic_utilix_config_discovery(self):
        import configparser

        cfg = configparser.ConfigParser()
        cfg.add_section("xrootd")
        cfg.set("xrootd", "redirector_url", "root://custom-origin.uchicago.edu/")
        cfg.set("xrootd", "subpath", "xenon/custom_subpath")
        cfg.set("xrootd", "timeout", "45.0")

        frontend = straxen.XRootDFrontend(uconfig=cfg)
        self.assertEqual(frontend.redirector_url, "root://custom-origin.uchicago.edu")
        self.assertEqual(frontend.subpath, "xenon/custom_subpath")
        self.assertEqual(frontend.xrootd_kwargs.get("timeout"), 45)
        self.assertIsInstance(frontend.xrootd_kwargs.get("timeout"), int)

    def test_explicit_override_over_utilix(self):
        import configparser

        cfg = configparser.ConfigParser()
        cfg.add_section("xrootd")
        cfg.set("xrootd", "redirector_url", "root://uconfig-origin.uchicago.edu/")
        cfg.set("xrootd", "subpath", "xenon/uconfig_subpath")

        frontend = straxen.XRootDFrontend(
            redirector_url="root://explicit-override.uchicago.edu/",
            subpath="xenon/explicit_subpath",
            uconfig=cfg,
        )
        self.assertEqual(frontend.redirector_url, "root://explicit-override.uchicago.edu")
        self.assertEqual(frontend.subpath, "xenon/explicit_subpath")

    def test_scitoken_discovery_explicit(self):
        import tempfile

        # Explicit string
        info = straxen.discover_scitoken(explicit_token="explicit_jwt_token", sync_environ=False)
        self.assertTrue(info.is_valid)
        self.assertEqual(info.token, "explicit_jwt_token")
        self.assertEqual(info.source, "explicit:token")
        self.assertEqual(info.read_token(), "explicit_jwt_token")

        # Explicit file
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("file_jwt_token\n")
            tf_path = tf.name

        try:
            info_file = straxen.discover_scitoken(explicit_token_file=tf_path, sync_environ=False)
            self.assertTrue(info_file.is_valid)
            self.assertEqual(info_file.token_file, tf_path)
            self.assertEqual(info_file.source, "explicit:token_file")
            self.assertEqual(info_file.read_token(), "file_jwt_token")
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_scitoken_discovery_environment(self):
        import tempfile
        import unittest.mock

        # Test BEARER_TOKEN alone
        with unittest.mock.patch.dict(os.environ, {"BEARER_TOKEN": "bearer_jwt_123"}, clear=True):
            info = straxen.discover_scitoken(sync_environ=False)
            self.assertEqual(info.source, "env:BEARER_TOKEN")
            self.assertEqual(info.token, "bearer_jwt_123")

        # Test BEARER_TOKEN_FILE
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("file_bearer_token\n")
            tf_path = tf.name

        try:
            # When only BEARER_TOKEN_FILE exists
            with unittest.mock.patch.dict(os.environ, {"BEARER_TOKEN_FILE": tf_path}, clear=True):
                info = straxen.discover_scitoken(sync_environ=False)
                self.assertEqual(info.source, "env:BEARER_TOKEN_FILE")
                self.assertEqual(info.token_file, tf_path)
                self.assertEqual(info.read_token(), "file_bearer_token")

            # WLCG precedence: when both exist, BEARER_TOKEN takes precedence
            with unittest.mock.patch.dict(
                os.environ,
                {"BEARER_TOKEN": "precedence_token", "BEARER_TOKEN_FILE": tf_path},
                clear=True,
            ):
                info = straxen.discover_scitoken(sync_environ=False)
                self.assertEqual(info.source, "env:BEARER_TOKEN")
                self.assertEqual(info.token, "precedence_token")

            # Fallthrough: non-existent BEARER_TOKEN_FILE falls through to BEARER_TOKEN
            with unittest.mock.patch.dict(
                os.environ,
                {"BEARER_TOKEN_FILE": "/does/not/exist/token", "BEARER_TOKEN": "fallback_token"},
                clear=True,
            ):
                info = straxen.discover_scitoken(sync_environ=False)
                self.assertEqual(info.source, "env:BEARER_TOKEN")
                self.assertEqual(info.token, "fallback_token")
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

        # Test SCITOKEN
        with unittest.mock.patch.dict(os.environ, {"SCITOKEN": "scitoken_xyz"}, clear=True):
            info = straxen.discover_scitoken(sync_environ=False)
            self.assertEqual(info.source, "env:SCITOKEN")
            self.assertEqual(info.token, "scitoken_xyz")

    def test_scitoken_discovery_utilix_config(self):
        import configparser
        import tempfile
        import unittest.mock

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            cfg = configparser.ConfigParser()
            cfg.add_section("xrootd")

            with tempfile.NamedTemporaryFile("w", delete=False) as tf:
                tf.write("uconfig_file_token\n")
                tf_path = tf.name

            try:
                cfg.set("xrootd", "token_file", tf_path)
                info = straxen.discover_scitoken(uconfig=cfg, sync_environ=False)
                self.assertEqual(info.source, "uconfig:xrootd:token_file")
                self.assertEqual(info.token_file, tf_path)
                self.assertEqual(info.read_token(), "uconfig_file_token")

                # Inline token
                cfg.remove_option("xrootd", "token_file")
                cfg.set("xrootd", "token", "raw_inline_token_str")
                info_inline = straxen.discover_scitoken(uconfig=cfg, sync_environ=False)
                self.assertEqual(info_inline.source, "uconfig:xrootd:token")
                self.assertEqual(info_inline.token, "raw_inline_token_str")
            finally:
                if os.path.exists(tf_path):
                    os.remove(tf_path)

    def test_scitoken_discovery_wlcg_tmp_and_xdg(self):
        import tempfile
        import unittest.mock

        uid = os.getuid() if hasattr(os, "getuid") else 1000
        wlcg_tmp = f"/tmp/bt_u{uid}"

        # Test /tmp/bt_u<uid> using mocks to avoid writing to real system paths
        orig_isfile = os.path.isfile
        orig_getsize = os.path.getsize

        def mock_isfile(path):
            if path == wlcg_tmp:
                return True
            return orig_isfile(path)

        def mock_getsize(path):
            if path == wlcg_tmp:
                return 100
            return orig_getsize(path)

        mock_file = unittest.mock.mock_open(read_data="tmp_wlcg_token\n")
        with (
            unittest.mock.patch.dict(os.environ, {}, clear=True),
            unittest.mock.patch("os.path.isfile", side_effect=mock_isfile),
            unittest.mock.patch("os.path.getsize", side_effect=mock_getsize),
            unittest.mock.patch("builtins.open", mock_file),
        ):
            info = straxen.discover_scitoken(sync_environ=False)
            self.assertEqual(info.source, "wlcg:tmp")
            self.assertEqual(info.read_token(), "tmp_wlcg_token")

        # Test XDG_RUNTIME_DIR
        with tempfile.TemporaryDirectory() as tmp_dir:
            xdg_token = os.path.join(tmp_dir, f"bt_u{uid}")
            with open(xdg_token, "w") as f:
                f.write("xdg_wlcg_token\n")

            xdg_env = {"XDG_RUNTIME_DIR": tmp_dir}
            with unittest.mock.patch.dict(os.environ, xdg_env, clear=True):
                info = straxen.discover_scitoken(sync_environ=False)
                self.assertEqual(info.source, "wlcg:xdg_runtime_dir")
                self.assertEqual(info.read_token(), "xdg_wlcg_token")

    def test_scitoken_environ_synchronization(self):
        import tempfile
        import unittest.mock

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("sync_token\n")
            tf_path = tf.name

        try:
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("BEARER_TOKEN", "BEARER_TOKEN_FILE")
            }
            with unittest.mock.patch.dict(os.environ, clean_env, clear=True):
                self.assertNotIn("BEARER_TOKEN_FILE", os.environ)
                info = straxen.discover_scitoken(explicit_token_file=tf_path, sync_environ=True)
                self.assertEqual(os.environ.get("BEARER_TOKEN_FILE"), tf_path)
                self.assertEqual(info.token_file, tf_path)
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_frontend_dynamic_reload(self):
        import configparser

        cfg = configparser.ConfigParser()
        cfg.add_section("xrootd")
        cfg.set("xrootd", "redirector_url", "root://first-origin.org/")

        frontend = straxen.XRootDFrontend(uconfig=cfg)
        self.assertEqual(frontend.redirector_url, "root://first-origin.org")

        cfg.set("xrootd", "redirector_url", "root://second-origin.org/")
        frontend.reload_config()
        self.assertEqual(frontend.redirector_url, "root://second-origin.org")

    def test_xenonnt_context_token_wiring(self):
        st = straxen.contexts.xenonnt(
            include_xrootd=True,
            _xrootd_url=self.redirector,
            _xrootd_subpath=self.subpath.lstrip("/"),
            _xrootd_token="context_bearer_token",
            _database_init=False,
        )
        xrootd_frontends = [sf for sf in st.storage if isinstance(sf, straxen.XRootDFrontend)]
        self.assertEqual(len(xrootd_frontends), 1)
        fe = xrootd_frontends[0]
        self.assertEqual(fe.token_info.token, "context_bearer_token")
        self.assertEqual(os.environ.get("BEARER_TOKEN"), "context_bearer_token")
        self.assertNotIn("token", fe.xrootd_kwargs)

    def test_missing_data_does_not_bump_failure_count(self):
        frontend = straxen.XRootDFrontend(
            redirector_urls=["memory://pool1", "memory://pool2"],
            subpath="processed",
        )
        missing_key = strax.DataKey(
            run_id="999999",
            data_type="records",
            lineage={"records": ["RecordsPlugin", "0.0.0", {}]},
        )
        with self.assertRaises(strax.DataNotAvailable):
            frontend.find(missing_key)

        pool = frontend.redirector_pool
        # FileNotFoundError on missing dataset should not increment failure counts
        for r in pool.redirectors:
            self.assertEqual(pool._failure_counts.get(r, 0), 0)
            self.assertEqual(pool._last_failure_time.get(r, 0.0), 0.0)

    def test_scitoken_explicit_overrides_stale_env(self):
        import unittest.mock

        with unittest.mock.patch.dict(
            os.environ,
            {"BEARER_TOKEN": "stale_token", "BEARER_TOKEN_FILE": "/path/to/stale"},
            clear=True,
        ):
            fe = straxen.XRootDFrontend(
                redirector_url="memory://",
                token="fresh_explicit_jwt",
            )
            self.assertEqual(fe.token_info.token, "fresh_explicit_jwt")
            self.assertEqual(os.environ.get("BEARER_TOKEN"), "fresh_explicit_jwt")
            self.assertNotIn("BEARER_TOKEN_FILE", os.environ)
            self.assertNotIn("token", fe.xrootd_kwargs)

    def test_incomplete_data_resolution(self):
        self._write_chunk_and_metadata(metadata_type="temp")
        frontend = straxen.XRootDFrontend(
            redirector_url=self.redirector,
            subpath=self.subpath.lstrip("/"),
        )
        # allow_incomplete=False raises DataNotAvailable
        with self.assertRaises(strax.DataNotAvailable):
            frontend.find(self.key, allow_incomplete=False)

        # allow_incomplete=True finds data and points backend key to _temp
        backend_name, found_key = frontend.find(self.key, allow_incomplete=True)
        self.assertEqual(backend_name, "XRootDBackend")
        self.assertTrue(found_key.endswith("_temp"))

        # Context can load incomplete data
        st = strax.Context(storage=[frontend])
        st.register(DummyPlugin)
        arr = st.get_array(self.run_id, self.data_type, allow_incomplete=True)
        self.assertEqual(len(arr), len(self.data))

    def test_timeout_int_conversion(self):
        frontend = straxen.XRootDFrontend(
            redirector_url="memory://",
            xrootd_kwargs={"timeout": 60.0},
        )
        self.assertEqual(frontend.xrootd_kwargs.get("timeout"), 60)
        self.assertIsInstance(frontend.xrootd_kwargs.get("timeout"), int)
        self.assertEqual(frontend.backends[0].xrootd_kwargs.get("timeout"), 60)
        self.assertIsInstance(frontend.backends[0].xrootd_kwargs.get("timeout"), int)
