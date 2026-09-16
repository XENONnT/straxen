import base64
import json
import os
import socket
import stat
import tempfile
import time
import unittest
import unittest.mock
import straxen
from straxen.scripts.validate_environment import main as cli_main
from straxen.storage.validator import (
    CheckResult,
    EnvironmentValidator,
    ValidationReport,
    parse_jwt_payload,
)


class TestEnvironmentValidator(unittest.TestCase):
    """Test suite for straxen Environment and Runtime Validator."""

    def test_parse_jwt_payload(self):
        # 1. Valid token
        header_json = json.dumps({"alg": "RS256"}).encode()
        header = base64.urlsafe_b64encode(header_json).decode().rstrip("=")
        payload_data = {
            "sub": "user@uchicago.edu",
            "exp": 1800000000,
            "iss": "https://wlcg.tokens.org",
        }
        payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).decode().rstrip("=")
        sig = "fake_signature_hash"
        token = f"{header}.{payload}.{sig}"

        parsed = parse_jwt_payload(token)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["sub"], "user@uchicago.edu")
        self.assertEqual(parsed["exp"], 1800000000)

        # 2. Invalid tokens
        self.assertIsNone(parse_jwt_payload("invalid_token"))
        self.assertIsNone(parse_jwt_payload("a.b"))
        self.assertIsNone(parse_jwt_payload("a.not_valid_b64!.c"))

    def test_check_python_packages(self):
        validator = EnvironmentValidator()
        checks = validator.check_python_packages()
        names = [c.name for c in checks]

        self.assertIn("python_version", names)
        self.assertIn("pkg:strax", names)
        self.assertIn("pkg:straxen", names)
        self.assertIn("pkg:fsspec", names)

        # Check strax and straxen pass
        strax_check = next(c for c in checks if c.name == "pkg:strax")
        self.assertEqual(strax_check.status, "PASS")

        # Mock missing required package
        with unittest.mock.patch("importlib.import_module", side_effect=ImportError("No module")):
            checks_missing = validator.check_python_packages()
            fsspec_check = next(c for c in checks_missing if c.name == "pkg:fsspec")
            self.assertEqual(fsspec_check.status, "FAIL")

    def test_check_system_binaries(self):
        validator = EnvironmentValidator()

        # Mock xrdfs and sbatch existing
        def fake_which(cmd):
            if cmd in ("xrdfs", "sbatch"):
                return f"/usr/bin/{cmd}"
            return None

        with unittest.mock.patch("shutil.which", side_effect=fake_which):
            checks = validator.check_system_binaries()
            xrdfs_check = next(c for c in checks if c.name == "bin:xrdfs")
            self.assertEqual(xrdfs_check.status, "PASS")
            self.assertIn("/usr/bin/xrdfs", xrdfs_check.message)

            xrdcp_check = next(c for c in checks if c.name == "bin:xrdcp")
            self.assertEqual(xrdcp_check.status, "WARN")

        # Container detection via env
        with unittest.mock.patch.dict(os.environ, {"APPTAINER_NAME": "xenonnt.sif"}, clear=False):
            checks_cont = validator.check_system_binaries()
            c_check = next(c for c in checks_cont if c.name == "container_runtime")
            self.assertEqual(c_check.status, "PASS")
            self.assertIn("xenonnt.sif", c_check.message)

    def test_check_auth_tokens(self):
        validator = EnvironmentValidator()

        # 1. No token
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            none_info = straxen.SciTokenInfo(source="none")
            with unittest.mock.patch("straxen.discover_scitoken", return_value=none_info):
                checks = validator.check_auth_tokens()
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0].status, "WARN")

        # 2. Valid unexpired token in file with 0600 permissions
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            now_ts = int(time.time())
            h_bytes = json.dumps({"alg": "RS256"}).encode()
            header = base64.urlsafe_b64encode(h_bytes).decode().rstrip("=")
            payload_data = {"sub": "test", "exp": now_ts + 7200}  # 2 hours
            p_bytes = json.dumps(payload_data).encode()
            payload = base64.urlsafe_b64encode(p_bytes).decode().rstrip("=")
            token_str = f"{header}.{payload}.sig"
            tf.write(token_str)
            tf_path = tf.name

        try:
            os.chmod(tf_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
            token_info = straxen.SciTokenInfo(token_file=tf_path, source="explicit:token_file")

            with unittest.mock.patch("straxen.discover_scitoken", return_value=token_info):
                checks = validator.check_auth_tokens()
                status_dict = {c.name: c.status for c in checks}
                self.assertEqual(status_dict["auth:scitoken_discovery"], "PASS")
                self.assertEqual(status_dict["auth:file_permissions"], "PASS")
                self.assertEqual(status_dict["auth:token_expiration"], "PASS")

            # 3. Test open file permissions (0644)
            os.chmod(tf_path, 0o644)
            with unittest.mock.patch("straxen.discover_scitoken", return_value=token_info):
                checks = validator.check_auth_tokens()
                perm_check = next(c for c in checks if c.name == "auth:file_permissions")
                self.assertEqual(perm_check.status, "WARN")

            # 4. Test expired token
            with open(tf_path, "w") as f:
                exp_data = json.dumps({"exp": now_ts - 100}).encode()
                payload_exp = base64.urlsafe_b64encode(exp_data).decode().rstrip("=")
                f.write(f"{header}.{payload_exp}.sig")

            with unittest.mock.patch("straxen.discover_scitoken", return_value=token_info):
                checks = validator.check_auth_tokens()
                exp_check = next(c for c in checks if c.name == "auth:token_expiration")
                self.assertEqual(exp_check.status, "FAIL")
        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)

    def test_check_cluster_mounts(self):
        with tempfile.TemporaryDirectory() as existing_dir:
            non_existent = os.path.join(existing_dir, "missing_subdir")
            validator = EnvironmentValidator(cluster_paths=[existing_dir, non_existent])
            checks = validator.check_cluster_mounts()

            self.assertEqual(len(checks), 2)
            c1 = next(c for c in checks if c.name == f"mount:{existing_dir}")
            self.assertEqual(c1.status, "PASS")

            c2 = next(c for c in checks if c.name == f"mount:{non_existent}")
            self.assertEqual(c2.status, "WARN")

    def test_check_xrootd_connectivity(self):
        validator = EnvironmentValidator(xrootd_url="root://test-server.org:1094")

        # 1. Success mock
        mock_conn = unittest.mock.MagicMock()
        with unittest.mock.patch("socket.create_connection", return_value=mock_conn):
            checks = validator.check_xrootd_connectivity()
            tcp_check = next(c for c in checks if c.name == "network:xrootd_tcp")
            self.assertEqual(tcp_check.status, "PASS")

        # 2. Timeout mock
        with unittest.mock.patch(
            "socket.create_connection", side_effect=socket.timeout("Connection timed out")
        ):
            checks_fail = validator.check_xrootd_connectivity()
            tcp_check = next(c for c in checks_fail if c.name == "network:xrootd_tcp")
            self.assertEqual(tcp_check.status, "WARN")

    def test_validation_report_aggregation(self):
        # HEALTHY
        r_healthy = ValidationReport(
            checks=[CheckResult("a", "PASS", "ok"), CheckResult("b", "PASS", "ok")]
        )
        self.assertEqual(r_healthy.overall_status, "HEALTHY")

        # DEGRADED
        r_degraded = ValidationReport(
            checks=[CheckResult("a", "PASS", "ok"), CheckResult("b", "WARN", "warning")]
        )
        self.assertEqual(r_degraded.overall_status, "DEGRADED")

        # UNHEALTHY
        r_unhealthy = ValidationReport(checks=[CheckResult("a", "FAIL", "error")])
        self.assertEqual(r_unhealthy.overall_status, "UNHEALTHY")

        # JSON export
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = os.path.join(tmp_dir, "report.json")
            r_degraded.to_json(json_file)
            self.assertTrue(os.path.isfile(json_file))

            with open(json_file, "r") as f:
                data = json.load(f)
            self.assertEqual(data["overall_status"], "DEGRADED")
            self.assertEqual(data["summary"]["warn"], 1)

    def test_cli_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_json = os.path.join(tmp_dir, "cli_report.json")

            # Normal run with skip network and skip mounts
            ret = cli_main(["--skip-network", "--skip-mounts", "--output", out_json])
            self.assertIn(ret, [0, 1])
            self.assertTrue(os.path.isfile(out_json))

            # JSON mode
            ret_json = cli_main(["--skip-network", "--skip-mounts", "--json"])
            self.assertIn(ret_json, [0, 1])


if __name__ == "__main__":
    unittest.main()
