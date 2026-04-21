#!/usr/bin/env python3
"""Verify the active JAX backend without relying on system CUDA paths."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DeviceInfo:
    """Serializable device information for verification output."""

    platform: str
    id: str
    kind: str


@dataclass
class VerificationReport:
    """Top-level verification report."""

    datarax_backend: str | None
    jax_platforms: str | None
    platform: str
    python: str
    jax_import_ok: bool
    jax_version: str | None
    default_backend: str | None
    gpu_device_count: int
    devices: list[DeviceInfo]
    error: str | None


def emit(message: str) -> None:
    """Write a single line to stdout."""
    sys.stdout.write(f"{message}\n")


@contextlib.contextmanager
def suppress_process_stderr():
    """Temporarily redirect process stderr to avoid noisy plugin-init logs."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w", encoding="utf-8") as null_stream:
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(null_stream.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


def collect_report() -> VerificationReport:
    """Collect backend information from the active environment."""
    base_report = VerificationReport(
        datarax_backend=os.environ.get("DATARAX_BACKEND"),
        jax_platforms=os.environ.get("JAX_PLATFORMS"),
        platform=platform.platform(),
        python=sys.version.split()[0],
        jax_import_ok=False,
        jax_version=None,
        default_backend=None,
        gpu_device_count=0,
        devices=[],
        error=None,
    )

    try:
        with suppress_process_stderr():
            import jax
    except (ImportError, OSError, RuntimeError) as exc:
        base_report.error = str(exc)
        return base_report

    try:
        with suppress_process_stderr():
            devices = [
                DeviceInfo(
                    platform=device.platform,
                    id=str(device),
                    kind=getattr(device, "device_kind", "unknown"),
                )
                for device in jax.devices()
            ]
            default_backend = jax.default_backend()
    except RuntimeError as exc:
        base_report.jax_import_ok = True
        base_report.jax_version = getattr(jax, "__version__", None)
        base_report.error = str(exc)
        return base_report

    gpu_device_count = sum(device.platform in {"gpu", "cuda"} for device in devices)
    return VerificationReport(
        datarax_backend=base_report.datarax_backend,
        jax_platforms=base_report.jax_platforms,
        platform=base_report.platform,
        python=base_report.python,
        jax_import_ok=True,
        jax_version=jax.__version__,
        default_backend=default_backend,
        gpu_device_count=gpu_device_count,
        devices=devices,
        error=None,
    )


def render_human_report(report: VerificationReport) -> str:
    """Render the report for terminal output."""
    lines = [
        "Datarax JAX backend verification",
        f"Platform: {report.platform}",
        f"Python: {report.python}",
        f"Configured Datarax backend: {report.datarax_backend or 'unset'}",
        f"JAX_PLATFORMS: {report.jax_platforms or 'unset'}",
    ]

    if not report.jax_import_ok:
        lines.append(f"JAX import failed: {report.error}")
        return "\n".join(lines)

    lines.extend(
        [
            f"JAX version: {report.jax_version}",
            f"Default backend: {report.default_backend}",
            f"GPU devices visible to JAX: {report.gpu_device_count}",
            "Devices:",
        ]
    )

    for device in report.devices:
        lines.append(f"  - {device.platform}: {device.kind} ({device.id})")

    if report.error:
        lines.append(f"Backend query failed: {report.error}")

    lines.append(
        "Note: Datarax leaves JAX_PLATFORMS unset by default so JAX can pick"
        " GPU when available and CPU otherwise."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit non-zero unless JAX can see at least one GPU device",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the verification report as JSON",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the verification script."""
    args = parse_args()
    report = collect_report()

    if args.json:
        payload: dict[str, Any] = asdict(report)
        emit(json.dumps(payload, indent=2, sort_keys=True))
    else:
        emit(render_human_report(report))

    if not report.jax_import_ok:
        return 1
    if report.error:
        return 1
    if args.require_gpu and report.gpu_device_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
