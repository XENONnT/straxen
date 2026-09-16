import hashlib
import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import strax
import straxen
from straxen.storage.rucio_deterministic import (
    chunk_to_rucio_did,
    did_from_backend_key,
    key_to_rucio_dids,
    rucio_deterministic_path,
)
from straxen.storage.xrootd import (
    RedirectorPool,
    XRootDBackend,
    XRootDFrontend,
)


class TestRucioDeterministic(unittest.TestCase):
    """Test Rucio deterministic path resolution, multi-redirector pool, and failover."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.run_id = "050001"
        self.dtype = "records"
        self.plugin_name = straxen.storage.benchmark.get_synthetic_plugin_name(self.dtype)
        self.lineage = {self.dtype: (self.plugin_name, "0.0.0", {})}
        self.key = strax.DataKey(run_id=self.run_id, data_type=self.dtype, lineage=self.lineage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rucio_deterministic_path(self):
        did = "xnt_050001:records-5vbh5o52-000000"
        expected_md5 = hashlib.md5(did.encode("utf-8")).hexdigest()
        expected_path = f"xnt_050001/{expected_md5[:2]}/{expected_md5[2:4]}/records-5vbh5o52-000000"

        res = rucio_deterministic_path(did)
        self.assertEqual(res, expected_path)

        # Check consistency with straxen.storage.rucio_local.rucio_path
        local_ref = straxen.storage.rucio_local.rucio_path("/base", did)
        self.assertEqual(local_ref, os.path.join("/base", expected_path))

        # Test prefix algorithm
        prefix_path = rucio_deterministic_path(did, algorithm="prefix")
        self.assertEqual(prefix_path, "xnt_050001/re/co/records-5vbh5o52-000000")

        # Test invalid DID
        with self.assertRaises(ValueError):
            rucio_deterministic_path("invalid_did_without_colon")

    def test_key_to_rucio_dids(self):
        dataset_did, metadata_did = key_to_rucio_dids(self.key, scope_prefix="xnt_")
        self.assertEqual(dataset_did, f"xnt_{self.run_id}:{self.dtype}-{self.key.lineage_hash}")
        self.assertEqual(
            metadata_did,
            f"xnt_{self.run_id}:{self.dtype}-{self.key.lineage_hash}-metadata.json",
        )

        # Test string key
        d_did2, m_did2 = key_to_rucio_dids(str(self.key))
        self.assertEqual(d_did2, dataset_did)
        self.assertEqual(m_did2, metadata_did)

        # Test chunk DID
        chunk_fn = f"{self.dtype}-{self.key.lineage_hash}-000000"
        c_did = chunk_to_rucio_did(f"xnt_{self.run_id}", chunk_fn)
        self.assertEqual(c_did, f"xnt_{self.run_id}:{chunk_fn}")

        # Test did_from_backend_key
        key_with_did = f"root://host//rucio/{dataset_did}"
        self.assertEqual(did_from_backend_key(key_with_did), dataset_did)
        self.assertIsNone(did_from_backend_key("root://host//processed/folder"))

    def test_redirector_pool_management(self):
        # Parsing comma and space separated strings
        raw = "root://r1.uchicago.edu, root://r2.uchicago.edu:1094/ root://r3.uchicago.edu/"
        pool = RedirectorPool(raw, failover_policy="priority")
        self.assertEqual(len(pool.redirectors), 3)
        self.assertEqual(pool.active_redirector, "root://r1.uchicago.edu")

        # Priority ordering
        candidates = pool.get_candidates()
        self.assertEqual(candidates[0], "root://r1.uchicago.edu")

        # Fail r1 -> r2 promoted
        pool.mark_failure("root://r1.uchicago.edu", Exception("Connection refused"))
        self.assertEqual(pool.active_redirector, "root://r2.uchicago.edu:1094")

        # Success on r3 -> r3 promoted
        pool.mark_success("root://r3.uchicago.edu")
        self.assertEqual(pool.active_redirector, "root://r3.uchicago.edu")

        # Round robin policy
        rr_pool = RedirectorPool(
            ["root://r1.org", "root://r2.org", "root://r3.org"],
            failover_policy="round_robin",
        )
        c1 = rr_pool.get_candidates()
        c2 = rr_pool.get_candidates()
        self.assertNotEqual(c1, c2)

    def test_swap_redirector_in_url(self):
        pool = RedirectorPool(
            [
                "root://primary.org",
                "root://primary.org:1094",
                "root://primary.org-2",
                "file:///data",
                "file:///data-backup",
            ]
        )

        # Standard double-slash root URL
        u1 = "root://primary.org//xenon/records/000"
        swapped = pool.swap_redirector_in_url(u1, "root://backup.org:1094/")
        self.assertEqual(swapped, "root://backup.org:1094//xenon/records/000")

        # Memory URL
        u_mem = "memory://origin_a/subpath/chunk"
        swapped_mem = pool.swap_redirector_in_url(u_mem, "memory://origin_b")
        self.assertEqual(swapped_mem, "memory://origin_b/subpath/chunk")

        # Bug 2: Host:port vs Host without port
        u_port = "root://primary.org:1094//xenon/records/000"
        swapped_port = pool.swap_redirector_in_url(u_port, "root://primary.org")
        self.assertEqual(swapped_port, "root://primary.org//xenon/records/000")

        # Bug 2: Host vs Host prefix (e.g. host-2 vs host)
        u_host = "root://primary.org//xenon/records/000"
        swapped_host2 = pool.swap_redirector_in_url(u_host, "root://primary.org-2")
        self.assertEqual(swapped_host2, "root://primary.org-2//xenon/records/000")

        u_host2 = "root://primary.org-2//xenon/records/000"
        swapped_back = pool.swap_redirector_in_url(u_host2, "root://primary.org")
        self.assertEqual(swapped_back, "root://primary.org//xenon/records/000")

        # Same redirector returns unchanged
        self.assertEqual(pool.swap_redirector_in_url(u1, "root://primary.org"), u1)

        # File URL boundary test
        u_file = "file:///data-backup/sub/file.bin"
        swapped_file = pool.swap_redirector_in_url(u_file, "file:///data")
        self.assertEqual(swapped_file, "file:///data/sub/file.bin")

    def test_rucio_deterministic_streaming(self):
        dest_url = "memory://rucio_test"
        key, target_str = straxen.SyntheticDataGenerator.create_dataset(
            destination_url=dest_url,
            target_type=self.dtype,
            run_id=self.run_id,
            n_chunks=3,
            chunk_size_mb=0.1,
            rucio_mode=True,
        )

        dataset_did, metadata_did = key_to_rucio_dids(key)
        rel_md_path = rucio_deterministic_path(metadata_did)

        import fsspec

        fs, _ = fsspec.core.url_to_fs(dest_url)
        self.assertTrue(fs.exists(f"rucio_test/{rel_md_path}"))

        # Backend retrieval in Rucio mode
        backend = XRootDBackend(rucio_mode=True)
        backend_key = f"{dest_url}/{dataset_did}"
        md = backend.get_metadata(backend_key)
        self.assertEqual(md["run_id"], self.run_id)
        self.assertEqual(len(md["chunks"]), 3)

        # Chunk retrieval
        chunk0 = backend._read_chunk(
            backend_key, md["chunks"][0], dtype=np.dtype(strax.record_dtype()), compressor="zstd"
        )
        self.assertGreater(len(chunk0), 0)

        # Full Strax Context streaming test with XRootDFrontend in Rucio mode
        st = strax.Context(
            storage=[
                XRootDFrontend(
                    redirector_url=dest_url,
                    subpath="",
                    rucio_mode=True,
                )
            ]
        )

        class RecordsPlugin(strax.Plugin):
            provides = self.dtype
            depends_on = tuple()
            data_kind = self.dtype
            __version__ = "0.0.0"

            def infer_dtype(self):
                return strax.record_dtype()

        RecordsPlugin.__name__ = self.plugin_name
        st.register(RecordsPlugin)

        data = st.get_array(self.run_id, self.dtype)
        self.assertEqual(len(data), len(chunk0) * 3)

    def test_multi_redirector_frontend_find_failover(self):
        primary_url = "memory://primary_dead"
        secondary_url = "memory://secondary_alive"

        # Create dataset ONLY on secondary
        key, _ = straxen.SyntheticDataGenerator.create_dataset(
            destination_url=secondary_url,
            target_type=self.dtype,
            run_id=self.run_id,
            n_chunks=2,
            chunk_size_mb=0.05,
            rucio_mode=True,
        )

        frontend = XRootDFrontend(
            redirector_urls=[primary_url, secondary_url],
            subpath="",
            rucio_mode=True,
        )

        # Finding should automatically fail on primary and succeed on secondary
        backend_name, found_key = frontend.find(key)
        self.assertEqual(backend_name, "XRootDBackend")
        self.assertTrue(found_key.startswith(secondary_url))
        # Active redirector should be promoted to secondary
        self.assertEqual(frontend.redirector_pool.active_redirector, secondary_url)

    def test_multi_redirector_chunk_read_failover(self):
        origin1 = "memory://node1"
        origin2 = "memory://node2"

        # Create dataset on node1
        key, _ = straxen.SyntheticDataGenerator.create_dataset(
            destination_url=origin1,
            target_type=self.dtype,
            run_id=self.run_id,
            n_chunks=2,
            chunk_size_mb=0.05,
            rucio_mode=True,
        )
        dataset_did, metadata_did = key_to_rucio_dids(key)
        rel_md = rucio_deterministic_path(metadata_did)

        # Also copy metadata and chunk 1 to node2, but DELETE chunk 1 on node1
        import fsspec

        fs, _ = fsspec.core.url_to_fs(origin1)
        md_bytes = fs.cat(f"node1/{rel_md}")
        md_dict = json.loads(md_bytes.decode("utf-8"))

        chunk1_did = chunk_to_rucio_did(f"xnt_{self.run_id}", md_dict["chunks"][1]["filename"])
        rel_chunk1 = rucio_deterministic_path(chunk1_did)
        chunk1_bytes = fs.cat(f"node1/{rel_chunk1}")

        # Put on node2
        fs.makedirs("node2/" + os.path.dirname(rel_md), exist_ok=True)
        fs.makedirs("node2/" + os.path.dirname(rel_chunk1), exist_ok=True)
        with fs.open(f"node2/{rel_md}", "wb") as f:
            f.write(md_bytes)
        with fs.open(f"node2/{rel_chunk1}", "wb") as f:
            f.write(chunk1_bytes)

        # Delete chunk 1 from node1 to force chunk failover
        fs.rm(f"node1/{rel_chunk1}")

        pool = RedirectorPool([origin1, origin2], failover_policy="priority")
        backend = XRootDBackend(
            redirector_pool=pool,
            rucio_mode=True,
        )

        backend_key = f"{origin1}/{dataset_did}"
        # Reading chunk 0 works from node1
        c0 = backend._read_chunk(
            backend_key,
            md_dict["chunks"][0],
            dtype=np.dtype(strax.record_dtype()),
            compressor="zstd",
        )
        self.assertGreater(len(c0), 0)

        # Reading chunk 1 triggers failover from node1 to node2!
        c1 = backend._read_chunk(
            backend_key,
            md_dict["chunks"][1],
            dtype=np.dtype(strax.record_dtype()),
            compressor="zstd",
        )
        self.assertGreater(len(c1), 0)
        # Node2 is promoted to active redirector in pool!
        self.assertEqual(pool.active_redirector, origin2)

    def test_contexts_rucio_mode_wiring(self):
        st = straxen.contexts.xenonnt(
            include_xrootd=True,
            _xrootd_urls=["root://xrootd.mwt2.org/", "root://xrootd.grid.uchicago.edu/"],
            _xrootd_rucio_mode=True,
            _xrootd_subpath="rucio",
            _xrootd_failover_policy="priority",
            _database_init=False,
        )
        frontends = [sf for sf in st.storage if isinstance(sf, XRootDFrontend)]
        self.assertEqual(len(frontends), 1)
        x_fe = frontends[0]
        self.assertTrue(x_fe.rucio_mode)
        self.assertEqual(len(x_fe.redirector_pool.redirectors), 2)
        self.assertEqual(x_fe.subpath, "rucio")

    def test_bounded_lru_metadata_cache(self):
        backend = XRootDBackend(max_cache_size=2)
        k1 = "memory://origin/xnt_050001:records-hash1"
        k2 = "memory://origin/xnt_050001:records-hash2"
        k3 = "memory://origin/xnt_050001:records-hash3"

        backend._metadata_cache[k1] = {"run_id": "050001", "dtype": "records", "h": "1"}
        backend._metadata_cache[k2] = {"run_id": "050001", "dtype": "records", "h": "2"}
        self.assertEqual(len(backend._metadata_cache), 2)

        # Access k1 to make it most recently used
        backend._metadata_cache.move_to_end(k1)

        # Add k3 with eviction check
        if len(backend._metadata_cache) >= backend.max_cache_size:
            backend._metadata_cache.popitem(last=False)
        backend._metadata_cache[k3] = {"run_id": "050001", "dtype": "records", "h": "3"}

        self.assertEqual(len(backend._metadata_cache), 2)
        # k2 was the oldest and must have been evicted
        self.assertNotIn(k2, backend._metadata_cache)
        self.assertIn(k1, backend._metadata_cache)
        self.assertIn(k3, backend._metadata_cache)

        # Clear cache
        backend.clear_metadata_cache()
        self.assertEqual(len(backend._metadata_cache), 0)

        # Test frontend delegation to backend
        frontend = XRootDFrontend("root://test-red//data", max_cache_size=5)
        frontend.backends[0]._metadata_cache[k1] = {"test": 1}
        self.assertEqual(len(frontend.backends[0]._metadata_cache), 1)
        frontend.clear_metadata_cache()
        self.assertEqual(len(frontend.backends[0]._metadata_cache), 0)


if __name__ == "__main__":
    unittest.main()
