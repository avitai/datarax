"""Automated Vast.ai A100 benchmark orchestration.

Single-command workflow:
1) provision cluster (SkyPilot),
2) verify remote hardware/backend,
3) run subset then full benchmark passes,
4) validate backend truth from result manifests,
5) download artifacts + optional analysis,
6) tear down cluster unless explicitly retained.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmarks.automation._orchestrator_utils import (
    _build_click_pin_hint,
    _build_sky_upgrade_hint,
    _confirm,
    _effective_allowed_gpu_tokens,
    _extract_json_line,
    _extract_nvidia_smi_rows,
    _is_known_skypilot_click_backend_bug,
    _normalize_capture_limit_chars,
    _primary_gpu_resource,
    _probe_python_package_version,
    _probe_skypilot_capabilities,
    _progress_bar,
    _remote_shell_prefix,
    _repo_root,
    _status,
    _timestamp_id,
    CommandResult,
    DEFAULT_ADAPTERS,
    DEFAULT_CAPTURE_LIMIT_CHARS,
    DEFAULT_SUBSET_SCENARIOS,
    OrchestrationError,
    REMOTE_DATA_ROOT,
    REMOTE_GPU_ENV_EXPORTS,
    REMOTE_RESULTS_ROOT,
)
from benchmarks.automation._subprocess import _run_logged_command


def preflight_checks(
    *,
    repo_root: Path,
    download_root: Path,
    template_path: Path,
    infra: str,
) -> dict[str, Any]:
    """Validate local prerequisites before launching cloud resources."""
    sky_executable = shutil.which("sky")
    if sky_executable is None:
        fallback_sky = repo_root / ".venv" / "bin" / "sky"
        if fallback_sky.exists():
            sky_executable = str(fallback_sky)
            _status(
                f"`sky` not found in PATH, falling back to virtualenv binary: {sky_executable}",
                level="WARN",
            )
        else:
            raise OrchestrationError("`sky` executable not found in PATH.")

    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise OrchestrationError(f"Missing virtualenv python: {venv_python}")

    if not template_path.exists():
        raise OrchestrationError(f"Template config not found: {template_path}")

    download_root.mkdir(parents=True, exist_ok=True)
    probe = download_root / ".write_test"
    probe.write_text("ok")
    probe.unlink()

    sky_caps = _probe_skypilot_capabilities(sky_executable)
    sky_supports_rsync = bool(sky_caps.get("supports_rsync"))
    sky_version = str(sky_caps.get("version", "unknown"))
    sky_upgrade_hint = _build_sky_upgrade_hint(repo_root=repo_root, infra=infra)
    click_pin_hint = _build_click_pin_hint(repo_root=repo_root)
    click_version = _probe_python_package_version(
        python_executable=venv_python,
        package="click",
    )

    if _is_known_skypilot_click_backend_bug(sky_version=sky_version, click_version=click_version):
        raise OrchestrationError(
            (
                "Detected incompatible SkyPilot/Click combination that causes "
                "`sky launch` to fail with `ValueError: False backend is not supported.` "
                f"(SkyPilot: {sky_version}; Click: {click_version}). "
                f"Fix by upgrading SkyPilot: {sky_upgrade_hint} "
                f"or pinning Click: {click_pin_hint}"
            ),
        )

    artifact_transfer_method = "sky-rsync"
    if not sky_supports_rsync:
        if shutil.which("scp") is None:
            raise OrchestrationError(
                "SkyPilot CLI does not expose `sky rsync` and `scp` is not available. "
                f"Upgrade SkyPilot or install scp first. Suggested command: {sky_upgrade_hint}",
            )
        artifact_transfer_method = "scp"
        _status(
            (
                f"SkyPilot {sky_version} does not expose `sky rsync`; "
                "artifact download will use scp compatibility mode"
            ),
            level="WARN",
        )
        _status(f"SkyPilot upgrade command: {sky_upgrade_hint}", level="WARN")

    return {
        "repo_root": str(repo_root),
        "venv_python": str(venv_python),
        "template": str(template_path),
        "sky_executable": sky_executable,
        "sky_version": sky_version,
        "sky_supports_rsync": sky_supports_rsync,
        "artifact_transfer_method": artifact_transfer_method,
        "sky_upgrade_hint": sky_upgrade_hint,
        "click_version": click_version,
        "click_pin_hint": click_pin_hint,
        "wandb_api_key_set": bool(os.getenv("WANDB_API_KEY")),
    }


def generate_sky_yaml(
    *,
    template_path: Path,
    output_root: Path,
    run_id: str,
    mode: str,
    on_demand: bool,
    cluster_name: str,
    filename: str = "generated_sky_config.yaml",
) -> Path:
    """Generate a run-specific SkyPilot config from the GPU template."""
    config = yaml.safe_load(template_path.read_text())
    config["name"] = cluster_name
    config.setdefault("resources", {})
    config["resources"]["accelerators"] = "A100:1"
    config["resources"]["use_spot"] = not on_demand

    envs = config.setdefault("envs", {})
    # SkyPilot rejects null env values; always emit a string value.
    envs["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
    envs["JAX_PLATFORMS"] = "cuda,cpu"
    envs["JAX_PLATFORM_NAME"] = "gpu"
    envs["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    envs["DATARAX_BENCH_RUN_ID"] = run_id
    envs["DATARAX_BENCH_MODE"] = mode

    sanity_cmd = (
        ".venv/bin/python -c "
        '"import jax; '
        "print(f'JAX {jax.__version__}, backend: {jax.default_backend()}, "
        "devices: {jax.devices()}')\" "
        f"| tee {REMOTE_RESULTS_ROOT}/logs/launch_sanity.log"
    )

    config["run"] = "\n".join(
        [
            *_remote_shell_prefix(),
            (
                f"mkdir -p {REMOTE_RESULTS_ROOT}/subset "
                f"{REMOTE_RESULTS_ROOT}/full {REMOTE_RESULTS_ROOT}/logs"
            ),
            sanity_cmd,
            "echo 'cluster-ready'",
        ],
    )

    yaml_path = output_root / filename
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path


def launch_cluster(
    *,
    yaml_path: Path,
    infra: str,
    cluster_name: str,
    logs_dir: Path,
    sky_executable: str = "sky",
    launch_timeout_sec: int | None = None,
    stall_timeout_sec: int | None = None,
    log_name: str = "sky_launch.log",
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
    sky_upgrade_hint: str | None = None,
    click_pin_hint: str | None = None,
) -> None:
    """Launch a SkyPilot cluster from a generated YAML config."""
    try:
        _run_logged_command(
            [
                sky_executable,
                "launch",
                str(yaml_path),
                "--infra",
                infra,
                "-c",
                cluster_name,
                "-y",
            ],
            logs_dir / log_name,
            check=True,
            timeout_sec=launch_timeout_sec,
            live_peek=live_peek,
            peek_interval_sec=peek_interval_sec,
            capture_limit_chars=capture_limit_chars,
            stall_timeout_sec=stall_timeout_sec,
            stall_diagnostics=(
                (
                    lambda: _run_stall_diagnostics(
                        cluster=cluster_name,
                        logs_dir=logs_dir,
                        sky_executable=sky_executable,
                        capture_limit_chars=capture_limit_chars,
                    )
                )
                if stall_timeout_sec is not None and stall_timeout_sec > 0
                else None
            ),
        )
    except OrchestrationError as exc:
        message = str(exc)
        if "False backend is not supported." in message:
            hints: list[str] = []
            if sky_upgrade_hint:
                hints.append(f"upgrade SkyPilot: {sky_upgrade_hint}")
            if click_pin_hint:
                hints.append(f"pin Click: {click_pin_hint}")
            if not hints:
                hints = ['upgrade SkyPilot (`pip install -U "skypilot[vast]"`)']
            hint_text = "; or ".join(hints)
            raise OrchestrationError(
                (
                    f"{message}\nDetected SkyPilot backend parsing incompatibility. "
                    f"Recommended fix: {hint_text}"
                ),
            ) from exc
        raise


def exec_remote(
    *,
    cluster: str,
    command: str,
    logs_dir: Path,
    log_name: str,
    check: bool = True,
    sky_executable: str = "sky",
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    gpus: str | None = None,
    secret_env_vars: list[str] | None = None,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> CommandResult:
    """Execute a command on a remote SkyPilot cluster."""
    args = [sky_executable, "exec", cluster]
    if gpus:
        args.extend(["--gpus", gpus])
    for env_var in sorted(set(secret_env_vars or [])):
        args.extend(["--secret", env_var])
    args.extend(["--", command])
    return _run_logged_command(
        args,
        logs_dir / log_name,
        check=check,
        live_peek=live_peek,
        peek_interval_sec=peek_interval_sec,
        capture_limit_chars=capture_limit_chars,
    )


def teardown_cluster(
    *,
    cluster: str,
    logs_dir: Path,
    sky_executable: str = "sky",
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> None:
    """Tear down a SkyPilot cluster."""
    _run_logged_command(
        [sky_executable, "down", cluster, "-y"],
        logs_dir / "sky_down.log",
        check=False,
        live_peek=live_peek,
        peek_interval_sec=peek_interval_sec,
        capture_limit_chars=capture_limit_chars,
    )


def _run_stall_diagnostic_command(
    *,
    args_candidates: list[list[str]],
    logs_dir: Path,
    base_log_name: str,
    sky_executable: str,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> CommandResult:
    """Run diagnostic command candidates until one succeeds."""
    last_result: CommandResult | None = None
    for idx, args in enumerate(args_candidates):
        log_name = f"{base_log_name}.log" if idx == 0 else f"{base_log_name}.fallback_{idx}.log"
        normalized = [sky_executable if token == "__SKY__" else token for token in args]  # nosec B105
        try:
            result = _run_logged_command(
                normalized,
                logs_dir / log_name,
                check=False,
                timeout_sec=60,
                live_peek=False,
                capture_limit_chars=capture_limit_chars,
            )
        except OrchestrationError as exc:
            result = CommandResult(
                args=normalized,
                returncode=1,
                stdout="",
                stderr=str(exc),
            )
        last_result = result
        if result.returncode == 0:
            return result
    if last_result is None:
        return CommandResult(args=[], returncode=1, stdout="", stderr="")
    return last_result


def _run_stall_diagnostics(
    *,
    cluster: str,
    logs_dir: Path,
    sky_executable: str,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> dict[str, str]:
    """Collect queue/log diagnostics when a long-running command stalls."""
    _status("Collecting stall diagnostics (sky queue + sky logs --tail)", level="WARN")

    queue_result = _run_stall_diagnostic_command(
        args_candidates=[
            ["__SKY__", "queue", cluster],
            ["__SKY__", "queue", "-c", cluster],
            ["__SKY__", "queue"],
        ],
        logs_dir=logs_dir,
        base_log_name="stall_sky_queue",
        sky_executable=sky_executable,
        capture_limit_chars=capture_limit_chars,
    )
    logs_result = _run_stall_diagnostic_command(
        args_candidates=[
            ["__SKY__", "logs", cluster, "--tail", "200"],
            ["__SKY__", "logs", "-c", cluster, "--tail", "200"],
            ["__SKY__", "logs", cluster],
        ],
        logs_dir=logs_dir,
        base_log_name="stall_sky_logs_tail",
        sky_executable=sky_executable,
        capture_limit_chars=capture_limit_chars,
    )

    report = {
        "queue": queue_result.combined,
        "logs_tail": logs_result.combined,
    }
    (logs_dir / "stall_diagnostics.json").write_text(json.dumps(report, indent=2))
    _status(f"Stall diagnostics saved: {logs_dir / 'stall_diagnostics.json'}", level="WARN")
    return report


def _is_cluster_visible(
    *,
    cluster: str,
    logs_dir: Path,
    sky_executable: str,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> bool:
    """Return True if cluster appears in `sky status --refresh` output."""
    status = _run_logged_command(
        [sky_executable, "status", cluster, "--refresh"],
        logs_dir / "sky_status_check.log",
        check=False,
        timeout_sec=60,
        live_peek=False,
        capture_limit_chars=capture_limit_chars,
    )
    text = status.combined
    lowered = text.lower()
    if "cluster(s) not found" in lowered:
        return False
    return cluster in text


def verify_remote_hardware(
    *,
    cluster: str,
    logs_dir: Path,
    allowed_gpu_tokens: list[str],
    sky_executable: str = "sky",
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> dict[str, Any]:
    """Verify the remote machine is A100-backed and JAX is on GPU."""
    normalized_tokens = _effective_allowed_gpu_tokens(allowed_gpu_tokens)
    gpu_resource = _primary_gpu_resource(normalized_tokens)
    gpu_info = exec_remote(
        cluster=cluster,
        command=("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        logs_dir=logs_dir,
        log_name="verify_nvidia_smi.log",
        sky_executable=sky_executable,
        live_peek=live_peek,
        peek_interval_sec=peek_interval_sec,
        gpus=gpu_resource,
        capture_limit_chars=capture_limit_chars,
    )
    raw_gpu_lines = [line.strip() for line in gpu_info.combined.splitlines() if line.strip()]
    gpu_lines = _extract_nvidia_smi_rows(gpu_info.combined)
    if not gpu_lines:
        raise OrchestrationError(
            f"No parseable GPU info returned by nvidia-smi. Raw output lines: {raw_gpu_lines!r}",
        )
    first_gpu = gpu_lines[0]
    first_gpu_lower = first_gpu.lower()
    if not any(token.lower() in first_gpu_lower for token in normalized_tokens):
        raise OrchestrationError(
            f"GPU check failed: expected one of {normalized_tokens}, got {first_gpu!r}",
        )

    probe_cmd = " && ".join(
        [
            "cd ~/datarax",
            *REMOTE_GPU_ENV_EXPORTS,
            '.venv/bin/python -c "import json,jax; '
            "print(json.dumps({'backend': jax.default_backend(), "
            "'devices': [str(d) for d in jax.devices()]}))\"",
        ],
    )
    jax_probe = exec_remote(
        cluster=cluster,
        command=probe_cmd,
        logs_dir=logs_dir,
        log_name="verify_jax_backend.log",
        sky_executable=sky_executable,
        live_peek=live_peek,
        peek_interval_sec=peek_interval_sec,
        gpus=gpu_resource,
        capture_limit_chars=capture_limit_chars,
    )
    payload = _extract_json_line(jax_probe.combined)
    backend = payload.get("backend")
    devices = payload.get("devices", [])
    if backend != "gpu":
        raise OrchestrationError(f"Expected JAX backend='gpu', got {backend!r}")
    if not any("cuda" in str(d).lower() for d in devices):
        raise OrchestrationError(f"Expected CUDA devices in JAX probe, got {devices!r}")

    report = {
        "ok": True,
        "gpu_info_lines": gpu_lines,
        "gpu_info_raw_lines": raw_gpu_lines,
        "jax_probe": payload,
    }
    (logs_dir / "hardware_verification.json").write_text(json.dumps(report, indent=2))
    return report


def _build_remote_bench_command(
    *,
    stage: str,
    repetitions: int,
    scenarios: list[str] | None,
    use_wandb: bool = True,
) -> str:
    output_dir = f"{REMOTE_RESULTS_ROOT}/{stage}"
    stage_log = f"{REMOTE_RESULTS_ROOT}/logs/{stage}.log"

    args = [
        ".venv/bin/python",
        "-m",
        "benchmarks.cli",
        "run",
        "--platform",
        "gpu",
        "--profile",
        "gpu_a100",
        "--repetitions",
        str(repetitions),
        "--output-dir",
        output_dir,
        "--data",
        REMOTE_DATA_ROOT,
    ]
    for adapter in DEFAULT_ADAPTERS:
        args.extend(["--adapters", adapter])
    if scenarios:
        for scenario in scenarios:
            args.extend(["--scenarios", scenario])
    if not use_wandb:
        args.append("--no-wandb")

    bench_cmd = " ".join(shlex.quote(token) for token in args)
    return "\n".join(
        [
            *_remote_shell_prefix(),
            f"mkdir -p {REMOTE_RESULTS_ROOT}/logs",
            f"{bench_cmd} 2>&1 | tee {stage_log}",
        ],
    )


def collect_artifacts(
    *,
    cluster: str,
    run_root: Path,
    logs_dir: Path,
    sky_executable: str = "sky",
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    transfer_method: str = "auto",
    sky_upgrade_hint: str | None = None,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> None:
    """Download benchmark results and remote logs."""

    def normalize_nested_results_dir() -> None:
        """Flatten `target/results/*` into `target/*` when scp adds an extra level."""
        nested_root = target / "results"
        if not nested_root.is_dir():
            return
        _status(
            "Normalizing nested artifact layout produced by scp (results/results -> results)",
            level="WARN",
        )
        for child in list(nested_root.iterdir()):
            destination = target / child.name
            if child.is_dir():
                shutil.copytree(child, destination, dirs_exist_ok=True)
                shutil.rmtree(child)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, destination)
                child.unlink()
        shutil.rmtree(nested_root, ignore_errors=True)

    def copy_with_scp() -> None:
        # SkyPilot CLI compatibility path for versions without `sky rsync`.
        _run_logged_command(
            [sky_executable, "status", cluster],
            logs_dir / "sky_status_for_scp.log",
            check=False,
            live_peek=False,
            capture_limit_chars=capture_limit_chars,
        )
        attempts = [
            {"remote_path": "~/results", "legacy_protocol": False},
            {"remote_path": "/root/results", "legacy_protocol": False},
            {"remote_path": "~/results/.", "legacy_protocol": False},
            {"remote_path": "~/results", "legacy_protocol": True},
        ]
        failures: list[str] = []
        for idx, attempt in enumerate(attempts):
            remote_path = attempt["remote_path"]
            legacy_protocol = bool(attempt["legacy_protocol"])
            scp_cmd = [
                "scp",
                "-r",
                "-q",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
            if legacy_protocol:
                scp_cmd.append("-O")
            scp_cmd.extend([f"{cluster}:{remote_path}", str(target)])

            log_name = "scp_results.log" if idx == 0 else f"scp_results_retry_{idx}.log"
            try:
                _run_logged_command(
                    scp_cmd,
                    logs_dir / log_name,
                    check=True,
                    timeout_sec=600,
                    live_peek=live_peek,
                    peek_interval_sec=peek_interval_sec,
                    capture_limit_chars=capture_limit_chars,
                )
                normalize_nested_results_dir()
                return
            except OrchestrationError as exc:
                failure = (
                    f"attempt={idx + 1} source={cluster}:{remote_path} "
                    f"legacy_protocol={legacy_protocol} error={exc}"
                )
                failures.append(failure)
                _status(
                    f"SCP artifact copy attempt {idx + 1}/{len(attempts)} failed; retrying",
                    level="WARN",
                )

        raise OrchestrationError(
            "All scp artifact copy attempts failed.\n" + "\n".join(failures[-4:]),
        )

    normalized_transfer = transfer_method.strip().lower().replace("_", "-")
    if normalized_transfer not in {"auto", "sky-rsync", "scp"}:
        raise OrchestrationError(
            "Invalid artifact transfer method: "
            f"{transfer_method!r}. Expected one of: auto, sky-rsync, scp",
        )

    target = run_root / "results"
    target.mkdir(parents=True, exist_ok=True)
    if normalized_transfer == "scp":
        copy_with_scp()
        return

    try:
        _run_logged_command(
            [sky_executable, "rsync", f"{cluster}:~/results/", str(target)],
            logs_dir / "sky_rsync_results.log",
            check=True,
            live_peek=live_peek,
            peek_interval_sec=peek_interval_sec,
            capture_limit_chars=capture_limit_chars,
        )
        return
    except OrchestrationError as exc:
        message = str(exc).lower()
        if "no such command 'rsync'" not in message:
            raise
        if normalized_transfer == "sky-rsync":
            upgrade_note = ""
            if sky_upgrade_hint:
                upgrade_note = f"\nSuggested upgrade command: {sky_upgrade_hint}"
            raise OrchestrationError(
                "Artifact transfer mode is forced to sky-rsync, but this SkyPilot CLI "
                "does not expose `sky rsync`."
                f"{upgrade_note}\nOr rerun with --artifact-transfer scp"
            ) from exc
        _status(
            "sky rsync is unavailable in this SkyPilot version; falling back to scp",
            level="WARN",
        )
    copy_with_scp()


def validate_results(
    *,
    stage: str,
    stage_results_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Validate manifest/result backend truth for one benchmark stage."""
    errors: list[str] = []
    samples_checked: list[str] = []

    manifest_path = stage_results_dir / "manifest.json"
    if not manifest_path.exists():
        report = {
            "stage": stage,
            "ok": False,
            "errors": [f"Missing manifest: {manifest_path}"],
            "samples_checked": [],
        }
        report_path.write_text(json.dumps(report, indent=2))
        return report

    manifest = json.loads(manifest_path.read_text())
    if "requested_platform" not in manifest:
        errors.append("manifest missing required field: requested_platform")
    if "active_backend" not in manifest:
        errors.append("manifest missing required field: active_backend")

    requested_platform = manifest.get("requested_platform")
    active_backend = manifest.get("active_backend")
    devices = manifest.get("environment", {}).get("platform", {}).get("devices", [])

    if requested_platform != "gpu":
        errors.append(f"requested_platform must be 'gpu', got {requested_platform!r}")
    if active_backend != "gpu":
        errors.append(f"active_backend must be 'gpu', got {active_backend!r}")
    if not any("cuda" in str(device).lower() for device in devices):
        errors.append(f"manifest devices missing CUDA: {devices!r}")

    adapter_files: list[str] = []
    for files in manifest.get("adapters", {}).values():
        adapter_files.extend(files)
    if not adapter_files:
        errors.append(
            "Manifest contains no result files; benchmark stage produced no adapter outputs.",
        )
    for fname in adapter_files[:3]:
        result_path = stage_results_dir / fname
        if not result_path.exists():
            errors.append(f"Missing result file listed in manifest: {fname}")
            continue
        data = json.loads(result_path.read_text())
        env = data.get("metadata", {}).get("environment", {})
        backend = env.get("platform", {}).get("backend")
        rdevices = env.get("platform", {}).get("devices", [])
        samples_checked.append(fname)
        if backend != "gpu":
            errors.append(f"{fname}: metadata backend={backend!r}, expected 'gpu'")
        if not any("cuda" in str(device).lower() for device in rdevices):
            errors.append(f"{fname}: metadata devices missing CUDA: {rdevices!r}")

    report = {
        "stage": stage,
        "ok": not errors,
        "manifest_path": str(manifest_path),
        "requested_platform": requested_platform,
        "active_backend": active_backend,
        "devices": devices,
        "samples_checked": samples_checked,
        "errors": errors,
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report


def run_stage(
    *,
    cluster: str,
    run_root: Path,
    logs_dir: Path,
    stage: str,
    repetitions: int,
    scenarios: list[str] | None,
    sky_executable: str = "sky",
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    artifact_transfer_method: str = "auto",
    sky_upgrade_hint: str | None = None,
    gpu_resource: str = "A100:1",
    use_wandb: bool = True,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> dict[str, Any]:
    """Execute one remote benchmark stage and validate downloaded artifacts."""
    command = _build_remote_bench_command(
        stage=stage,
        repetitions=repetitions,
        scenarios=scenarios,
        use_wandb=use_wandb,
    )
    secret_env_vars: list[str] = []
    if use_wandb:
        secret_env_vars.append("WANDB_API_KEY")
    exec_remote(
        cluster=cluster,
        command=command,
        logs_dir=logs_dir,
        log_name=f"run_{stage}.log",
        sky_executable=sky_executable,
        live_peek=live_peek,
        peek_interval_sec=peek_interval_sec,
        gpus=gpu_resource,
        secret_env_vars=secret_env_vars,
        capture_limit_chars=capture_limit_chars,
    )

    collect_artifacts(
        cluster=cluster,
        run_root=run_root,
        logs_dir=logs_dir,
        sky_executable=sky_executable,
        live_peek=live_peek,
        peek_interval_sec=peek_interval_sec,
        transfer_method=artifact_transfer_method,
        sky_upgrade_hint=sky_upgrade_hint,
        capture_limit_chars=capture_limit_chars,
    )
    return validate_results(
        stage=stage,
        stage_results_dir=run_root / "results" / stage,
        report_path=run_root / f"validation_report_{stage}.json",
    )


def run_analysis(
    *,
    repo_root: Path,
    run_root: Path,
    stages: list[str],
    logs_dir: Path,
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> None:
    """Run local analysis for completed stages."""
    python = repo_root / ".venv" / "bin" / "python"
    for stage in stages:
        stage_dir = run_root / "results" / stage
        if not stage_dir.exists():
            continue
        _run_logged_command(
            [
                str(python),
                "-m",
                "benchmarks.cli",
                "analyze",
                "--results-dir",
                str(stage_dir),
                "--output",
                str(run_root / "analysis" / stage),
            ],
            logs_dir / f"analyze_{stage}.log",
            check=True,
            cwd=repo_root,
            live_peek=live_peek,
            peek_interval_sec=peek_interval_sec,
            capture_limit_chars=capture_limit_chars,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _OrchestrationContext:
    """Immutable run context shared across orchestration helpers."""

    args: argparse.Namespace
    repo_root: Path
    run_id: str
    run_root: Path
    logs_dir: Path
    template_path: Path
    capture_limit_chars: int | None
    allowed_gpu_tokens: list[str]


def _plan_steps(args: argparse.Namespace) -> list[str]:
    """Compute the ordered list of orchestration steps for the given arguments."""
    steps = ["preflight", "generate_config"]
    if not args.dry_run:
        steps.extend(["launch", "verify_hardware"])
        if args.mode in {"subset", "two-pass"}:
            steps.append("subset")
        if args.mode in {"full", "two-pass"}:
            steps.append("full")
        if args.analyze:
            steps.append("analysis")
    steps.append("finalize")
    return steps


def _run_preflight_and_resolve_config(
    ctx: _OrchestrationContext,
    mark_step: Callable[[str, str], None],
) -> tuple[dict[str, Any], str, str | None, str | None, bool]:
    """Run preflight checks and resolve artifact transfer + wandb settings.

    Returns:
        Tuple of (preflight_dict, artifact_transfer_method, sky_upgrade_hint,
        click_pin_hint, use_remote_wandb).
    """
    mark_step("preflight", "Running preflight checks")
    preflight = preflight_checks(
        repo_root=ctx.repo_root,
        download_root=ctx.run_root,
        template_path=ctx.template_path,
        infra=ctx.args.infra,
    )
    (ctx.logs_dir / "preflight.json").write_text(json.dumps(preflight, indent=2))
    _status("Preflight checks passed")
    if not preflight.get("wandb_api_key_set"):
        _status(
            "WANDB_API_KEY is not set in current shell; remote W&B export will be disabled",
            level="WARN",
        )
    preflight_transfer_method = preflight.get("artifact_transfer_method", "auto")
    configured_transfer_method = ctx.args.artifact_transfer
    artifact_transfer_method = (
        preflight_transfer_method
        if configured_transfer_method == "auto"
        else configured_transfer_method
    )
    sky_upgrade_hint = preflight.get("sky_upgrade_hint")
    click_pin_hint = preflight.get("click_pin_hint")
    use_remote_wandb = bool(preflight.get("wandb_api_key_set"))
    _status(
        f"Artifact transfer mode: {artifact_transfer_method} "
        f"(preflight={preflight_transfer_method}, user={configured_transfer_method})",
    )
    if not use_remote_wandb:
        _status(
            "Remote benchmark stages will run with --no-wandb (WANDB_API_KEY is not configured).",
            level="WARN",
        )
    return preflight, artifact_transfer_method, sky_upgrade_hint, click_pin_hint, use_remote_wandb


def _handle_dry_run(
    ctx: _OrchestrationContext,
    preflight: dict[str, Any],
    sky_yaml: Path,
    mark_step: Callable[[str, str], None],
) -> int:
    """Finalize a dry run: write summary and return exit code."""
    mark_step("finalize", "Dry run finalization")
    summary = {
        "run_id": ctx.run_id,
        "mode": ctx.args.mode,
        "infra": ctx.args.infra,
        "cluster": ctx.args.cluster,
        "on_demand": ctx.args.on_demand,
        "effective_on_demand": ctx.args.on_demand,
        "dry_run": True,
        "ok": True,
        "run_root": str(ctx.run_root),
        "sky_config": str(sky_yaml),
        "preflight": preflight,
    }
    (ctx.run_root / "validation_report.json").write_text(json.dumps(summary, indent=2))
    _status("Dry run complete. No cloud resources were provisioned.")
    _status(f"Dry-run summary: {ctx.run_root / 'validation_report.json'}")
    return 0


def _retry_launch_on_visible_cluster(
    ctx: _OrchestrationContext,
    sky_yaml: Path,
    sky_executable: str,
    sky_upgrade_hint: str | None,
    click_pin_hint: str | None,
) -> bool:
    """Retry on-demand launch if the cluster is already visible after a timeout.

    Returns True if the retry succeeded.
    """
    _status(
        "On-demand launch timed out; checking cluster visibility before fallback",
        level="WARN",
    )
    if not _is_cluster_visible(
        cluster=ctx.args.cluster,
        logs_dir=ctx.logs_dir,
        sky_executable=sky_executable,
        capture_limit_chars=ctx.capture_limit_chars,
    ):
        return False

    _status(
        "Cluster is visible in Sky status; retrying on-demand launch on same cluster",
        level="WARN",
    )
    try:
        launch_cluster(
            yaml_path=sky_yaml,
            infra=ctx.args.infra,
            cluster_name=ctx.args.cluster,
            logs_dir=ctx.logs_dir,
            sky_executable=sky_executable,
            launch_timeout_sec=ctx.args.launch_timeout_sec,
            stall_timeout_sec=ctx.args.stall_timeout_sec,
            log_name="sky_launch_retry.log",
            live_peek=ctx.args.live_peek,
            peek_interval_sec=ctx.args.peek_interval_sec,
            capture_limit_chars=ctx.capture_limit_chars,
            sky_upgrade_hint=sky_upgrade_hint,
            click_pin_hint=click_pin_hint,
        )
    except OrchestrationError:
        return False
    return True


def _launch_spot_fallback(
    ctx: _OrchestrationContext,
    sky_executable: str,
    sky_upgrade_hint: str | None,
    click_pin_hint: str | None,
    fallback_exc: OrchestrationError,
) -> tuple[Path, bool]:
    """Attempt spot-instance fallback after an on-demand launch failure.

    Returns the (sky_yaml, launched) tuple. Raises OrchestrationError if the
    user declines or the fallback itself fails.
    """
    _status(
        f"On-demand launch failed; fallback available: {fallback_exc}",
        level="WARN",
    )
    if not _confirm("Retry launch with spot instances?", assume_yes=ctx.args.yes):
        raise OrchestrationError(
            "On-demand launch failed and spot fallback was declined.",
        ) from fallback_exc

    sky_yaml = generate_sky_yaml(
        template_path=ctx.template_path,
        output_root=ctx.run_root,
        run_id=ctx.run_id,
        mode=ctx.args.mode,
        on_demand=False,
        cluster_name=ctx.args.cluster,
        filename="generated_sky_config_spot_fallback.yaml",
    )
    _status(f"Generated fallback spot Sky config: {sky_yaml}", level="WARN")
    launch_cluster(
        yaml_path=sky_yaml,
        infra=ctx.args.infra,
        cluster_name=ctx.args.cluster,
        logs_dir=ctx.logs_dir,
        sky_executable=sky_executable,
        launch_timeout_sec=ctx.args.launch_timeout_sec,
        stall_timeout_sec=ctx.args.stall_timeout_sec,
        log_name="sky_launch_spot_fallback.log",
        live_peek=ctx.args.live_peek,
        peek_interval_sec=ctx.args.peek_interval_sec,
        capture_limit_chars=ctx.capture_limit_chars,
        sky_upgrade_hint=sky_upgrade_hint,
        click_pin_hint=click_pin_hint,
    )
    return sky_yaml, True


def _launch_cluster_with_fallbacks(
    ctx: _OrchestrationContext,
    sky_yaml: Path,
    sky_executable: str,
    sky_upgrade_hint: str | None,
    click_pin_hint: str | None,
) -> tuple[bool, bool]:
    """Launch cluster with timeout retry and optional spot fallback.

    Returns (used_spot_fallback, effective_on_demand_changed_to_spot).
    """
    try:
        launch_cluster(
            yaml_path=sky_yaml,
            infra=ctx.args.infra,
            cluster_name=ctx.args.cluster,
            logs_dir=ctx.logs_dir,
            sky_executable=sky_executable,
            launch_timeout_sec=ctx.args.launch_timeout_sec,
            stall_timeout_sec=ctx.args.stall_timeout_sec,
            live_peek=ctx.args.live_peek,
            peek_interval_sec=ctx.args.peek_interval_sec,
            capture_limit_chars=ctx.capture_limit_chars,
            sky_upgrade_hint=sky_upgrade_hint,
            click_pin_hint=click_pin_hint,
        )
        return False, False
    except OrchestrationError as exc:
        if "stall detected" in str(exc).lower():
            raise

        launch_ok = False
        # Timeout may happen after cloud allocation started; if the cluster is visible,
        # retry launch on the same cluster before opening fallback.
        if "timed out" in str(exc).lower() and ctx.args.on_demand:
            launch_ok = _retry_launch_on_visible_cluster(
                ctx,
                sky_yaml,
                sky_executable,
                sky_upgrade_hint,
                click_pin_hint,
            )

        if launch_ok:
            return False, False

        if ctx.args.on_demand and ctx.args.allow_spot_fallback:
            _launch_spot_fallback(
                ctx,
                sky_executable,
                sky_upgrade_hint,
                click_pin_hint,
                exc,
            )
            return True, True

        raise exc from None


def _run_benchmark_stages(
    ctx: _OrchestrationContext,
    *,
    reports: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
    artifact_transfer_method: str,
    use_remote_wandb: bool,
    sky_executable: str,
    mark_step: Callable[[str, str], None],
) -> tuple[str, list[str]]:
    """Execute subset/full benchmark stages and optional analysis.

    Returns (current_step, stages_to_analyze) so the caller can track progress.
    """
    current_step = "verify_hardware"
    stages_to_analyze: list[str] = []

    if ctx.args.mode in {"subset", "two-pass"}:
        current_step = "subset"
        mark_step("subset", "Running subset stage")
        reports["subset"] = run_stage(
            cluster=ctx.args.cluster,
            run_root=ctx.run_root,
            logs_dir=ctx.logs_dir,
            stage="subset",
            repetitions=ctx.args.subset_repetitions,
            scenarios=DEFAULT_SUBSET_SCENARIOS,
            sky_executable=sky_executable,
            live_peek=ctx.args.live_peek,
            peek_interval_sec=ctx.args.peek_interval_sec,
            artifact_transfer_method=artifact_transfer_method,
            sky_upgrade_hint=preflight.get("sky_upgrade_hint"),
            gpu_resource=_primary_gpu_resource(ctx.allowed_gpu_tokens),
            use_wandb=use_remote_wandb,
            capture_limit_chars=ctx.capture_limit_chars,
        )
        stages_to_analyze.append("subset")
        _status(
            f"Subset validation status: {'PASS' if reports['subset']['ok'] else 'FAIL'}",
        )
        if not reports["subset"]["ok"]:
            raise OrchestrationError(
                "Subset validation failed; skipping full run. "
                f"See {ctx.run_root / 'validation_report_subset.json'}",
            )

    if ctx.args.mode in {"full", "two-pass"}:
        current_step = "full"
        mark_step("full", "Running full stage")
        reports["full"] = run_stage(
            cluster=ctx.args.cluster,
            run_root=ctx.run_root,
            logs_dir=ctx.logs_dir,
            stage="full",
            repetitions=ctx.args.full_repetitions,
            scenarios=None,
            sky_executable=sky_executable,
            live_peek=ctx.args.live_peek,
            peek_interval_sec=ctx.args.peek_interval_sec,
            artifact_transfer_method=artifact_transfer_method,
            sky_upgrade_hint=preflight.get("sky_upgrade_hint"),
            gpu_resource=_primary_gpu_resource(ctx.allowed_gpu_tokens),
            use_wandb=use_remote_wandb,
            capture_limit_chars=ctx.capture_limit_chars,
        )
        stages_to_analyze.append("full")
        _status(
            f"Full validation status: {'PASS' if reports['full']['ok'] else 'FAIL'}",
        )
        if not reports["full"]["ok"]:
            raise OrchestrationError(
                f"Full validation failed. See {ctx.run_root / 'validation_report_full.json'}",
            )

    if ctx.args.analyze:
        current_step = "analysis"
        mark_step("analysis", "Running local analysis")
        run_analysis(
            repo_root=ctx.repo_root,
            run_root=ctx.run_root,
            stages=stages_to_analyze,
            logs_dir=ctx.logs_dir,
            live_peek=ctx.args.live_peek,
            peek_interval_sec=ctx.args.peek_interval_sec,
            capture_limit_chars=ctx.capture_limit_chars,
        )
        _status("Analysis completed")

    return current_step, stages_to_analyze


def _write_failure_report(
    ctx: _OrchestrationContext,
    *,
    exc: Exception,
    current_step: str,
    effective_on_demand: bool,
    used_spot_fallback: bool,
    reports: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
) -> None:
    """Persist a failure summary to disk and emit error status messages."""
    failure = {
        "run_id": ctx.run_id,
        "mode": ctx.args.mode,
        "infra": ctx.args.infra,
        "cluster": ctx.args.cluster,
        "on_demand": ctx.args.on_demand,
        "effective_on_demand": effective_on_demand,
        "used_spot_fallback": used_spot_fallback,
        "run_root": str(ctx.run_root),
        "step": current_step,
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "reports": reports,
        "preflight": preflight,
    }
    (ctx.run_root / "validation_report.json").write_text(json.dumps(failure, indent=2))
    _status(
        f"Run failed at step={current_step}: {type(exc).__name__}: {exc}",
        level="ERROR",
    )
    _status(f"Failure summary: {ctx.run_root / 'validation_report.json'}", level="ERROR")


def _teardown_if_needed(
    ctx: _OrchestrationContext,
    *,
    cluster_started: bool,
    preflight: dict[str, Any],
    primary_failure: Exception | None,
) -> None:
    """Tear down the cluster unless retention was requested."""
    if cluster_started and not ctx.args.keep_cluster:
        _status("Tearing down cluster")
        try:
            teardown_cluster(
                cluster=ctx.args.cluster,
                logs_dir=ctx.logs_dir,
                sky_executable=preflight.get("sky_executable", "sky"),
                live_peek=ctx.args.live_peek,
                peek_interval_sec=ctx.args.peek_interval_sec,
                capture_limit_chars=ctx.capture_limit_chars,
            )
        except Exception as teardown_exc:
            _status(f"Teardown failed: {teardown_exc}", level="ERROR")
            if primary_failure is None:
                raise
        else:
            _status("Teardown command finished")
    elif ctx.args.keep_cluster:
        _status("Cluster retained due to --keep-cluster", level="WARN")


def orchestrate(args: argparse.Namespace) -> int:
    """Top-level orchestration entry point."""
    repo_root = _repo_root()
    run_id = args.run_id or _timestamp_id()
    run_root = Path(args.download_dir).resolve() / run_id
    logs_dir = run_root / "orchestrator_logs"
    template_path = (repo_root / args.template).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    capture_limit_chars = _normalize_capture_limit_chars(args.capture_limit_chars)

    ctx = _OrchestrationContext(
        args=args,
        repo_root=repo_root,
        run_id=run_id,
        run_root=run_root,
        logs_dir=logs_dir,
        template_path=template_path,
        capture_limit_chars=capture_limit_chars,
        allowed_gpu_tokens=_effective_allowed_gpu_tokens(args.allowed_gpu),
    )

    _status(
        f"Starting run_id={run_id} infra={args.infra} cluster={args.cluster} "
        f"mode={args.mode} on_demand={args.on_demand} "
        f"launch_timeout_sec={args.launch_timeout_sec} "
        f"stall_timeout_sec={args.stall_timeout_sec} "
        f"capture_limit_chars={capture_limit_chars}",
    )
    _status(f"Run artifacts directory: {run_root}")
    _status(f"Detailed command logs directory: {logs_dir}")

    reports: dict[str, dict[str, Any]] = {}
    preflight: dict[str, Any] = {}
    cluster_started = False
    current_step = "preflight"
    primary_failure: Exception | None = None
    used_spot_fallback = False
    effective_on_demand = args.on_demand

    planned_steps = _plan_steps(args)
    total_steps = len(planned_steps)
    completed_steps = 0

    def mark_step(step_name: str, detail: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        bar = _progress_bar(completed_steps, total_steps)
        _status(f"{bar} Step {completed_steps}/{total_steps} ({step_name}): {detail}")

    try:
        (
            preflight,
            artifact_transfer_method,
            sky_upgrade_hint,
            click_pin_hint,
            use_remote_wandb,
        ) = _run_preflight_and_resolve_config(ctx, mark_step)

        current_step = "generate_config"
        mark_step("generate_config", "Generating SkyPilot config")
        sky_yaml = generate_sky_yaml(
            template_path=template_path,
            output_root=run_root,
            run_id=run_id,
            mode=args.mode,
            on_demand=args.on_demand,
            cluster_name=args.cluster,
        )
        _status(f"Generated Sky config: {sky_yaml}")
        sky_executable = preflight.get("sky_executable", "sky")

        if args.dry_run:
            return _handle_dry_run(ctx, preflight, sky_yaml, mark_step)

        if not _confirm(
            f"Proceed with provisioning cluster '{args.cluster}' on infra '{args.infra}'?",
            assume_yes=args.yes,
        ):
            raise OrchestrationError("Provisioning aborted by user confirmation.")

        current_step = "launch"
        mark_step("launch", "Launching cluster (this may take several minutes)")
        cluster_started = True
        spot_fallback, spot_changed = _launch_cluster_with_fallbacks(
            ctx,
            sky_yaml,
            sky_executable,
            sky_upgrade_hint,
            click_pin_hint,
        )
        if spot_fallback:
            used_spot_fallback = True
            effective_on_demand = False
            current_step = "launch_spot_fallback"
        _status("Cluster launch completed")

        current_step = "verify_hardware"
        mark_step("verify_hardware", "Verifying remote hardware/backend")
        hardware = verify_remote_hardware(
            cluster=args.cluster,
            logs_dir=logs_dir,
            allowed_gpu_tokens=ctx.allowed_gpu_tokens,
            sky_executable=sky_executable,
            live_peek=args.live_peek,
            peek_interval_sec=args.peek_interval_sec,
            capture_limit_chars=capture_limit_chars,
        )
        (run_root / "hardware_report.json").write_text(json.dumps(hardware, indent=2))
        _status("Remote hardware/backend verification passed")

        current_step, _ = _run_benchmark_stages(
            ctx,
            reports=reports,
            preflight=preflight,
            artifact_transfer_method=artifact_transfer_method,
            use_remote_wandb=use_remote_wandb,
            sky_executable=sky_executable,
            mark_step=mark_step,
        )

        current_step = "finalize"
        mark_step("finalize", "Writing validation summary")
        summary = {
            "run_id": run_id,
            "mode": args.mode,
            "infra": args.infra,
            "cluster": args.cluster,
            "on_demand": args.on_demand,
            "effective_on_demand": effective_on_demand,
            "used_spot_fallback": used_spot_fallback,
            "run_root": str(run_root),
            "reports": reports,
            "ok": all(report.get("ok", False) for report in reports.values()),
        }
        (run_root / "validation_report.json").write_text(json.dumps(summary, indent=2))
        _status("Run completed successfully")
        _status(f"Validation summary: {run_root / 'validation_report.json'}")
        return 0
    except Exception as exc:
        primary_failure = exc
        _write_failure_report(
            ctx,
            exc=exc,
            current_step=current_step,
            effective_on_demand=effective_on_demand,
            used_spot_fallback=used_spot_fallback,
            reports=reports,
            preflight=preflight,
        )
        raise
    finally:
        _teardown_if_needed(
            ctx,
            cluster_started=cluster_started,
            preflight=preflight,
            primary_failure=primary_failure,
        )


# Re-export CLI entry points from the _cli module for backward compatibility.
# Tests and __main__ callers reference these through ``vo.build_parser``.
from benchmarks.automation._cli import (
    build_parser as build_parser,  # noqa: E402, F811
    main as main,  # noqa: E402, F811
)


if __name__ == "__main__":
    main()
