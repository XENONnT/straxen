"""Environment and Runtime Diagnostic Validator for straxen and XRootD streaming.

Validates software dependencies, system CLI tools, SciToken / WLCG authentication,
cluster filesystem mount points, and remote XRootD redirector connectivity.
"""

import base64
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit
import strax
import straxen

export, __all__ = strax.exporter()
__all__.extend(
    [
        "CheckResult",
        "ValidationReport",
        "EnvironmentValidator",
    ]
)

log = logging.getLogger("straxen.validator")


def parse_jwt_payload(token: str) -> Optional[Dict[str, Any]]:
    """Decode JWT payload claims without third-party cryptographic libraries."""
    parts = token.strip().split(".")
    if len(parts) != 3:
        return None
    try:
        payload_b64 = parts[1]
        payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        return json.loads(decoded)
    except Exception as e:
        log.debug(f"Failed to parse JWT payload: {e}")
        return None


@export
@dataclass
class CheckResult:
    """Individual diagnostic check result."""

    name: str
    status: str  # "PASS", "WARN", "FAIL", "SKIP"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@export
@dataclass
class ValidationReport:
    """Aggregated environment diagnostic report."""

    checks: List[CheckResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hostname: str = field(default_factory=socket.getfqdn)
    platform_info: Dict[str, Any] = field(
        default_factory=lambda: {
            "python_version": sys.version.split()[0],
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }
    )

    @property
    def overall_status(self) -> str:
        """Compute overall system health: HEALTHY, DEGRADED, or UNHEALTHY."""
        if any(c.status == "FAIL" for c in self.checks):
            return "UNHEALTHY"
        if any(c.status == "WARN" for c in self.checks):
            return "DEGRADED"
        return "HEALTHY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "overall_status": self.overall_status,
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "platform_info": self.platform_info,
            "summary": {
                "total": len(self.checks),
                "pass": sum(1 for c in self.checks if c.status == "PASS"),
                "warn": sum(1 for c in self.checks if c.status == "WARN"),
                "fail": sum(1 for c in self.checks if c.status == "FAIL"),
                "skip": sum(1 for c in self.checks if c.status == "SKIP"),
            },
            "checks": [asdict(c) for c in self.checks],
        }

    def to_json(self, filepath: str) -> None:
        """Export report to JSON file."""
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown_table(self) -> str:
        """Format report into a Markdown table."""
        lines = [
            f"### Environment Diagnostic Report: **{self.overall_status}**",
            f"*Host:* `{self.hostname}` | *Python:* `{self.platform_info['python_version']}` "
            f"| *Time:* `{self.timestamp}`\n",
            "| Domain / Component | Status | Message |",
            "| --- | --- | --- |",
        ]

        status_symbols = {
            "PASS": "[PASS]",
            "WARN": "[WARN]",
            "FAIL": "[FAIL]",
            "SKIP": "[SKIP]",
        }

        for c in self.checks:
            badge = status_symbols.get(c.status, c.status)
            lines.append(f"| {c.name} | **{badge}** | {c.message} |")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print formatted ASCII/Markdown summary."""
        print("\n" + "=" * 80)
        print(f"            STRAXEN ENVIRONMENT VALIDATION: {self.overall_status}")
        print("=" * 80)
        print(self.to_markdown_table())
        print("=" * 80 + "\n")


@export
class EnvironmentValidator:
    """Diagnostic validator for straxen runtime and streaming infrastructure."""

    def __init__(
        self,
        xrootd_url: Optional[str] = None,
        cluster_paths: Optional[List[str]] = None,
    ):
        self.xrootd_url = xrootd_url or straxen.DEFAULT_REDIRECTOR_URL
        self.cluster_paths = cluster_paths or [
            "/project/lgrandi",
            "/project2/lgrandi",
            "/dali/lgrandi",
            "/scratch/midway3",
            "/tmp",
        ]

    def check_python_packages(self) -> List[CheckResult]:
        """Validate Python runtime and core package imports."""
        results: List[CheckResult] = []

        # Python version check
        py_ver = sys.version_info
        if py_ver >= (3, 10):
            results.append(
                CheckResult(
                    name="python_version",
                    status="PASS",
                    message=f"Python version {sys.version.split()[0]} >= 3.10",
                    details={"version": sys.version.split()[0]},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="python_version",
                    status="FAIL",
                    message=f"Python version {sys.version.split()[0]} is older than 3.10",
                    details={"version": sys.version.split()[0]},
                )
            )

        packages = [
            ("strax", True),
            ("straxen", True),
            ("fsspec", True),
            ("fsspec_xrootd", False),
            ("XRootD", False),
            ("utilix", False),
        ]

        for pkg_name, required in packages:
            try:
                mod = importlib.import_module(pkg_name)
                ver = getattr(mod, "__version__", "installed")
                results.append(
                    CheckResult(
                        name=f"pkg:{pkg_name}",
                        status="PASS",
                        message=f"{pkg_name} ({ver}) is installed and importable",
                        details={"version": ver},
                    )
                )
            except ImportError as e:
                status = "FAIL" if required else "WARN"
                msg = (
                    f"Required package {pkg_name} is missing: {e}"
                    if required
                    else f"Optional package {pkg_name} is not installed"
                )
                results.append(
                    CheckResult(
                        name=f"pkg:{pkg_name}",
                        status=status,
                        message=msg,
                        details={"error": str(e)},
                    )
                )

        return results

    def check_system_binaries(self) -> List[CheckResult]:
        """Verify presence of Grid and XRootD command-line tools."""
        results: List[CheckResult] = []

        binaries = [
            ("xrdfs", "XRootD filesystem query client", False),
            ("xrdcp", "XRootD file transfer utility", False),
            ("sbatch", "SLURM batch scheduler submission", False),
        ]

        for bname, desc, req in binaries:
            path = shutil.which(bname)
            if path:
                results.append(
                    CheckResult(
                        name=f"bin:{bname}",
                        status="PASS",
                        message=f"{bname} found at {path} ({desc})",
                        details={"path": path},
                    )
                )
            else:
                status = "FAIL" if req else "WARN"
                results.append(
                    CheckResult(
                        name=f"bin:{bname}",
                        status=status,
                        message=f"{bname} not found on PATH ({desc})",
                    )
                )

        # Container engine check
        apptainer_path = shutil.which("apptainer") or shutil.which("singularity")
        in_container = (
            os.path.exists("/.singularity.d")
            or "SINGULARITY_NAME" in os.environ
            or "APPTAINER_NAME" in os.environ
        )

        if in_container:
            c_name = os.environ.get("APPTAINER_NAME") or os.environ.get(
                "SINGULARITY_NAME", "container"
            )
            results.append(
                CheckResult(
                    name="container_runtime",
                    status="PASS",
                    message=f"Running inside container: {c_name}",
                    details={"in_container": True, "container_name": c_name},
                )
            )
        elif apptainer_path:
            results.append(
                CheckResult(
                    name="container_runtime",
                    status="PASS",
                    message=f"Container runtime available at {apptainer_path}",
                    details={"path": apptainer_path},
                )
            )
        else:
            results.append(
                CheckResult(
                    name="container_runtime",
                    status="WARN",
                    message="Neither running inside nor found apptainer/singularity on host",
                )
            )

        return results

    def check_auth_tokens(self) -> List[CheckResult]:
        """Validate SciToken / WLCG Bearer Token discovery, expiry, and permissions."""
        results: List[CheckResult] = []

        token_info = straxen.discover_scitoken(sync_environ=False)
        if not token_info.is_valid:
            results.append(
                CheckResult(
                    name="auth:scitoken",
                    status="WARN",
                    message=(
                        "No SciToken / WLCG Bearer Token discovered; "
                        "unauthenticated access only"
                    ),
                    details={"source": token_info.source},
                )
            )
            return results

        results.append(
            CheckResult(
                name="auth:scitoken_discovery",
                status="PASS",
                message=f"Discovered token via {token_info.source}",
                details={
                    "source": token_info.source,
                    "token_file": token_info.token_file,
                },
            )
        )

        # Inspect file permissions if file
        if token_info.token_file and os.path.exists(token_info.token_file):
            try:
                st_mode = os.stat(token_info.token_file).st_mode
                perms = oct(st_mode & 0o777)
                if st_mode & 0o077 != 0:
                    results.append(
                        CheckResult(
                            name="auth:file_permissions",
                            status="WARN",
                            message=f"Token file permissions are {perms}; recommend 0600",
                            details={"permissions": perms},
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            name="auth:file_permissions",
                            status="PASS",
                            message=f"Token file permissions are secure ({perms})",
                            details={"permissions": perms},
                        )
                    )
            except OSError as e:
                results.append(
                    CheckResult(
                        name="auth:file_permissions",
                        status="WARN",
                        message=f"Could not stat token file: {e}",
                    )
                )

        # Parse JWT payload if possible
        raw_token = token_info.read_token()
        if raw_token:
            payload = parse_jwt_payload(raw_token)
            if payload and "exp" in payload:
                exp_ts = payload["exp"]
                now_ts = time.time()
                remaining_s = exp_ts - now_ts
                exp_dt = datetime.fromtimestamp(exp_ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )

                if remaining_s <= 0:
                    results.append(
                        CheckResult(
                            name="auth:token_expiration",
                            status="FAIL",
                            message=f"Token expired at {exp_dt}",
                            details={"exp": exp_dt, "remaining_seconds": remaining_s},
                        )
                    )
                elif remaining_s < 1800:  # < 30 mins
                    mins = int(remaining_s / 60)
                    results.append(
                        CheckResult(
                            name="auth:token_expiration",
                            status="WARN",
                            message=f"Token expires soon in {mins} minutes ({exp_dt})",
                            details={"exp": exp_dt, "remaining_seconds": remaining_s},
                        )
                    )
                else:
                    hours = remaining_s / 3600.0
                    results.append(
                        CheckResult(
                            name="auth:token_expiration",
                            status="PASS",
                            message=f"Token valid for {hours:.1f} hours (expires {exp_dt})",
                            details={"exp": exp_dt, "remaining_seconds": remaining_s},
                        )
                    )

        return results

    def check_cluster_mounts(
        self, paths: Optional[List[str]] = None
    ) -> List[CheckResult]:
        """Verify accessibility of Midway and cluster storage paths."""
        results: List[CheckResult] = []
        target_paths = paths or self.cluster_paths

        for p in target_paths:
            if os.path.exists(p):
                readable = os.access(p, os.R_OK)
                writable = os.access(p, os.W_OK)
                perm_str = (
                    "read/write"
                    if (readable and writable)
                    else ("read-only" if readable else "inaccessible")
                )
                results.append(
                    CheckResult(
                        name=f"mount:{p}",
                        status="PASS",
                        message=f"Directory accessible ({perm_str}): {p}",
                        details={"path": p, "readable": readable, "writable": writable},
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name=f"mount:{p}",
                        status="WARN",
                        message=f"Cluster path not mounted or missing: {p}",
                        details={"path": p},
                    )
                )

        return results

    def check_xrootd_connectivity(
        self,
        url: Optional[Union[str, List[str]]] = None,
        timeout: float = 5.0,
    ) -> List[CheckResult]:
        """Verify network connectivity and handshake to one or more XRootD redirectors."""
        from .xrootd import parse_redirector_list

        results: List[CheckResult] = []
        if url:
            target_urls = parse_redirector_list(url)
        else:
            target_urls = parse_redirector_list(self.xrootd_url)

        if not target_urls:
            target_urls = ["root://midway-origin.uchicago.edu/"]

        multi = len(target_urls) > 1

        for target_url in target_urls:
            parsed = urlsplit(target_url)
            host = parsed.hostname or (
                target_url.split("://")[1].split("/")[0].split(":")[0]
                if "://" in target_url
                else target_url
            )
            port = parsed.port or 1094

            tcp_name = f"network:xrootd_tcp:{host}:{port}" if multi else "network:xrootd_tcp"

            # TCP socket check
            t0 = time.perf_counter()
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    results.append(
                        CheckResult(
                            name=tcp_name,
                            status="PASS",
                            message=(
                                f"TCP connection to {host}:{port} succeeded in {latency_ms:.1f} ms"
                            ),
                            details={"host": host, "port": port, "latency_ms": latency_ms},
                        )
                    )
            except (socket.timeout, OSError) as e:
                results.append(
                    CheckResult(
                        name=tcp_name,
                        status="WARN",
                        message=f"Cannot reach XRootD endpoint {host}:{port}: {e}",
                        details={"host": host, "port": port, "error": str(e)},
                    )
                )
                continue

            # Protocol ping via xrdfs if available
            if shutil.which("xrdfs"):
                t_ping = time.perf_counter()
                ping_name = (
                    f"network:xrootd_xrdfs_ping:{host}:{port}"
                    if multi
                    else "network:xrootd_xrdfs_ping"
                )
                try:
                    subprocess.run(
                        ["xrdfs", f"{host}:{port}", "ping"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=timeout,
                    )
                    ping_latency_ms = (time.perf_counter() - t_ping) * 1000.0
                    results.append(
                        CheckResult(
                            name=ping_name,
                            status="PASS",
                            message=(
                                f"xrdfs ping to {host}:{port} succeeded in {ping_latency_ms:.1f} ms"
                            ),
                            details={"host": host, "port": port, "latency_ms": ping_latency_ms},
                        )
                    )
                except (subprocess.SubprocessError, OSError) as e:
                    results.append(
                        CheckResult(
                            name=ping_name,
                            status="WARN",
                            message=f"xrdfs ping returned non-zero response: {e}",
                            details={"error": str(e)},
                        )
                    )

        return results

    def validate_all(
        self,
        skip_network: bool = False,
        skip_mounts: bool = False,
    ) -> ValidationReport:
        """Run all diagnostic checks and assemble final ValidationReport."""
        all_checks: List[CheckResult] = []

        all_checks.extend(self.check_python_packages())
        all_checks.extend(self.check_system_binaries())
        all_checks.extend(self.check_auth_tokens())

        if not skip_mounts:
            all_checks.extend(self.check_cluster_mounts())

        if not skip_network:
            all_checks.extend(self.check_xrootd_connectivity())

        return ValidationReport(checks=all_checks)
