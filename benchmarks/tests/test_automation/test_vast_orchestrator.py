"""Tests for Vast benchmark orchestrator automation."""

from __future__ import annotations

import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from benchmarks.automation import _orchestrator_utils as vo_utils, vast_orchestrator as vo


def _args(tmp_path: Path, **overrides) -> Namespace:
    base = {
        "infra": "vast",
        "cluster": "datarax-vast-a100",
        "mode": "two-pass",
        "download_dir": str(tmp_path / "downloads"),
        "template": "benchmarks/sky/gpu-benchmark.yaml",
        "subset_repetitions": 3,
        "full_repetitions": 5,
        "allowed_gpu": ["A100"],
        "analyze": False,
        "keep_cluster": False,
        "run_id": "test_run",
        "on_demand": True,
        "dry_run": False,
        "yes": True,
        "allow_spot_fallback": True,
        "launch_timeout_sec": 300,
        "stall_timeout_sec": 300,
        "capture_limit_chars": 1_000_000,
        "live_peek": True,
        "peek_interval_sec": 5,
        "artifact_transfer": "auto",
    }
    base.update(overrides)
    return Namespace(**base)


def _command_result(stdout: str = "", stderr: str = "", code: int = 0) -> vo.CommandResult:
    return vo.CommandResult(args=["cmd"], returncode=code, stdout=stdout, stderr=stderr)


class TestGenerateSkyYaml:
    def test_on_demand_sets_use_spot_false_and_gpu_env(self, tmp_path: Path):
        template = tmp_path / "template.yaml"
        template.write_text(
            yaml.safe_dump(
                {
                    "name": "old-name",
                    "resources": {"accelerators": "A100:1", "use_spot": True},
                    "envs": {"TF_CPP_MIN_LOG_LEVEL": "2"},
                    "setup": "echo setup",
                    "run": "echo old run",
                },
                sort_keys=False,
            ),
        )

        with patch.dict("os.environ", {"WANDB_API_KEY": "test-key"}, clear=False):
            out = vo.generate_sky_yaml(
                template_path=template,
                output_root=tmp_path / "out",
                run_id="run123",
                mode="two-pass",
                on_demand=True,
                cluster_name="new-cluster",
            )
        generated = yaml.safe_load(out.read_text())

        assert generated["name"] == "new-cluster"
        assert generated["resources"]["use_spot"] is False
        assert generated["resources"]["accelerators"] == "A100:1"
        assert generated["envs"]["WANDB_API_KEY"] == "test-key"
        assert generated["envs"]["JAX_PLATFORMS"] == "cuda,cpu"
        assert generated["envs"]["JAX_PLATFORM_NAME"] == "gpu"
        assert generated["envs"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert generated["envs"]["DATARAX_BENCH_RUN_ID"] == "run123"

    def test_wandb_key_defaults_to_empty_string(self, tmp_path: Path):
        template = tmp_path / "template.yaml"
        template.write_text(
            yaml.safe_dump(
                {
                    "name": "old-name",
                    "resources": {"accelerators": "A100:1", "use_spot": True},
                    "envs": {"WANDB_API_KEY": None},
                    "setup": "echo setup",
                    "run": "echo old run",
                },
                sort_keys=False,
            ),
        )
        with patch.dict("os.environ", {}, clear=True):
            out = vo.generate_sky_yaml(
                template_path=template,
                output_root=tmp_path / "out",
                run_id="run124",
                mode="two-pass",
                on_demand=True,
                cluster_name="new-cluster",
            )
        generated = yaml.safe_load(out.read_text())
        assert generated["envs"]["WANDB_API_KEY"] == ""


class TestPreflight:
    def test_preflight_fails_for_known_skypilot_click_incompatibility(self, tmp_path: Path):
        repo_root = tmp_path / "repo"
        venv_bin = repo_root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/usr/bin/env python\n")
        template = repo_root / "benchmarks" / "sky" / "gpu-benchmark.yaml"
        template.parent.mkdir(parents=True)
        template.write_text("name: test\nresources: {}\n")

        def _which(name: str) -> str | None:
            if name == "sky":
                return "/usr/bin/sky"
            if name == "scp":
                return "/usr/bin/scp"
            return None

        with (
            patch.object(vo.shutil, "which", side_effect=_which),
            patch.object(
                vo,
                "_probe_skypilot_capabilities",
                return_value={
                    "version": "skypilot, version 0.11.2",
                    "supports_rsync": True,
                },
            ),
            patch.object(vo, "_probe_python_package_version", return_value="8.3.1"),
        ):
            with pytest.raises(vo.OrchestrationError, match="False backend is not supported"):
                vo.preflight_checks(
                    repo_root=repo_root,
                    download_root=tmp_path / "download",
                    template_path=template,
                    infra="vast",
                )

    def test_preflight_uses_scp_transfer_when_sky_rsync_is_missing(self, tmp_path: Path):
        repo_root = tmp_path / "repo"
        venv_bin = repo_root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/usr/bin/env python\n")
        template = repo_root / "benchmarks" / "sky" / "gpu-benchmark.yaml"
        template.parent.mkdir(parents=True)
        template.write_text("name: test\nresources: {}\n")

        def _which(name: str) -> str | None:
            if name == "sky":
                return "/usr/bin/sky"
            if name == "scp":
                return "/usr/bin/scp"
            return None

        with (
            patch.object(vo.shutil, "which", side_effect=_which),
            patch.object(
                vo,
                "_probe_skypilot_capabilities",
                return_value={"version": "0.11.2", "supports_rsync": False},
            ),
        ):
            report = vo.preflight_checks(
                repo_root=repo_root,
                download_root=tmp_path / "download",
                template_path=template,
                infra="vast",
            )

        assert report["artifact_transfer_method"] == "scp"
        assert report["sky_supports_rsync"] is False
        assert report["sky_version"] == "0.11.2"
        assert "skypilot[vast]" in report["sky_upgrade_hint"]

    def test_preflight_fails_when_rsync_missing_and_scp_unavailable(self, tmp_path: Path):
        repo_root = tmp_path / "repo"
        venv_bin = repo_root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/usr/bin/env python\n")
        template = repo_root / "benchmarks" / "sky" / "gpu-benchmark.yaml"
        template.parent.mkdir(parents=True)
        template.write_text("name: test\nresources: {}\n")

        def _which(name: str) -> str | None:
            if name == "sky":
                return "/usr/bin/sky"
            return None

        with (
            patch.object(vo.shutil, "which", side_effect=_which),
            patch.object(
                vo,
                "_probe_skypilot_capabilities",
                return_value={"version": "0.11.2", "supports_rsync": False},
            ),
        ):
            with pytest.raises(vo.OrchestrationError, match="`scp` is not available"):
                vo.preflight_checks(
                    repo_root=repo_root,
                    download_root=tmp_path / "download",
                    template_path=template,
                    infra="vast",
                )


class TestSkyProbe:
    def test_probe_skypilot_capabilities_detects_rsync(self):
        def _fake_run(args, check, capture_output, text):  # noqa: ANN001
            del capture_output, check, text
            if "--help" in args:
                return subprocess.CompletedProcess(  # type: ignore[name-defined]
                    args=args,
                    returncode=0,
                    stdout="Commands:\n  launch\n  rsync\n  down\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(  # type: ignore[name-defined]
                args=args,
                returncode=0,
                stdout="skypilot, version 1.2.3",
                stderr="",
            )

        with patch.object(vo_utils.subprocess, "run", side_effect=_fake_run):
            report = vo._probe_skypilot_capabilities("/usr/bin/sky")  # noqa: SLF001

        assert report["supports_rsync"] is True
        assert "1.2.3" in report["version"]


class TestHardwareVerification:
    def test_verify_remote_hardware_success(self, tmp_path: Path):
        with patch.object(
            vo,
            "exec_remote",
            side_effect=[
                _command_result(stdout="NVIDIA A100-SXM4-80GB, 81920 MiB, 550.127"),
                _command_result(stdout='{"backend":"gpu","devices":["cuda:0"]}'),
            ],
        ):
            report = vo.verify_remote_hardware(
                cluster="cluster",
                logs_dir=tmp_path,
                allowed_gpu_tokens=["A100"],
            )
        assert report["ok"] is True
        assert report["jax_probe"]["backend"] == "gpu"

    def test_verify_remote_hardware_ignores_skypilot_wrapper_lines(self, tmp_path: Path):
        wrapped_nvidia_output = "\n".join(
            [
                "Submitting job to cluster: datarax-vast-a100",
                (
                    "Command to run: nvidia-smi --query-gpu=name,memory.total,"
                    "driver_version --format=csv,noheader"
                ),
                "(sky-cmd, pid=1966) NVIDIA A100-SXM4-80GB, 81920 MiB, 570.211.01",
            ],
        )
        with patch.object(
            vo,
            "exec_remote",
            side_effect=[
                _command_result(stdout=wrapped_nvidia_output),
                _command_result(stdout='{"backend":"gpu","devices":["cuda:0"]}'),
            ],
        ) as exec_remote_mock:
            report = vo.verify_remote_hardware(
                cluster="cluster",
                logs_dir=tmp_path,
                allowed_gpu_tokens=["A100"],
            )

        assert report["ok"] is True
        assert report["gpu_info_lines"][0].startswith("NVIDIA A100")
        first_call = exec_remote_mock.call_args_list[0].kwargs
        second_call = exec_remote_mock.call_args_list[1].kwargs
        assert first_call["gpus"] == "A100:1"
        assert second_call["gpus"] == "A100:1"

    def test_verify_remote_hardware_matches_allowed_gpu_case_insensitively(
        self,
        tmp_path: Path,
    ):
        with patch.object(
            vo,
            "exec_remote",
            side_effect=[
                _command_result(stdout="NVIDIA A100-SXM4-80GB, 81920 MiB, 550.127"),
                _command_result(stdout='{"backend":"gpu","devices":["cuda:0"]}'),
            ],
        ):
            report = vo.verify_remote_hardware(
                cluster="cluster",
                logs_dir=tmp_path,
                allowed_gpu_tokens=["a100"],
            )

        assert report["ok"] is True

    def test_extract_json_line_parses_prefixed_and_ansi_wrapped_json(self):
        text = '\x1b[36m(sky-cmd, pid=2735)\x1b[0m {"backend": "gpu", "devices": ["cuda:0"]}'
        payload = vo._extract_json_line(text)  # noqa: SLF001
        assert payload["backend"] == "gpu"
        assert payload["devices"] == ["cuda:0"]

    def test_verify_remote_hardware_fails_for_non_gpu_backend(self, tmp_path: Path):
        with patch.object(
            vo,
            "exec_remote",
            side_effect=[
                _command_result(stdout="NVIDIA A100-SXM4-80GB, 81920 MiB, 550.127"),
                _command_result(stdout='{"backend":"cpu","devices":["cpu:0"]}'),
            ],
        ):
            with pytest.raises(vo.OrchestrationError, match="Expected JAX backend='gpu'"):
                vo.verify_remote_hardware(
                    cluster="cluster",
                    logs_dir=tmp_path,
                    allowed_gpu_tokens=["A100"],
                )


class TestValidation:
    def test_validate_results_flags_backend_mismatch(self, tmp_path: Path):
        stage_dir = tmp_path / "results" / "subset"
        stage_dir.mkdir(parents=True)
        manifest = {
            "platform": "gpu",
            "requested_platform": "cpu",
            "active_backend": "cpu",
            "environment": {"platform": {"devices": ["cpu:0"]}},
            "adapters": {"Datarax": ["Datarax_CV-1_small.json"]},
        }
        (stage_dir / "manifest.json").write_text(json.dumps(manifest))
        result = {
            "metadata": {
                "environment": {
                    "platform": {"backend": "cpu", "devices": ["cpu:0"]},
                },
            },
        }
        (stage_dir / "Datarax_CV-1_small.json").write_text(json.dumps(result))

        report = vo.validate_results(
            stage="subset",
            stage_results_dir=stage_dir,
            report_path=tmp_path / "validation_subset.json",
        )

        assert report["ok"] is False
        assert any("requested_platform" in e for e in report["errors"])
        assert any("active_backend" in e for e in report["errors"])

    def test_validate_results_flags_missing_backend_truth_fields(self, tmp_path: Path):
        stage_dir = tmp_path / "results" / "subset"
        stage_dir.mkdir(parents=True)
        manifest = {
            "platform": "gpu",
            "environment": {"platform": {"devices": ["cuda:0"]}},
            "adapters": {},
        }
        (stage_dir / "manifest.json").write_text(json.dumps(manifest))

        report = vo.validate_results(
            stage="subset",
            stage_results_dir=stage_dir,
            report_path=tmp_path / "validation_subset.json",
        )

        assert report["ok"] is False
        assert any("missing required field" in e for e in report["errors"])

    def test_validate_results_fails_when_manifest_has_no_result_files(self, tmp_path: Path):
        stage_dir = tmp_path / "results" / "subset"
        stage_dir.mkdir(parents=True)
        manifest = {
            "platform": "gpu",
            "requested_platform": "gpu",
            "active_backend": "gpu",
            "environment": {"platform": {"devices": ["cuda:0"]}},
            "adapters": {"Datarax": [], "tf.data": []},
        }
        (stage_dir / "manifest.json").write_text(json.dumps(manifest))

        report = vo.validate_results(
            stage="subset",
            stage_results_dir=stage_dir,
            report_path=tmp_path / "validation_subset.json",
        )

        assert report["ok"] is False
        assert any("no result files" in e.lower() for e in report["errors"])


class TestCommandBuilders:
    def test_remote_benchmark_command_uses_module_cli(self):
        command = vo._build_remote_bench_command(  # noqa: SLF001
            stage="subset",
            repetitions=3,
            scenarios=["CV-1", "NLP-1"],
        )

        assert ".venv/bin/python -m benchmarks.cli run" in command
        assert ".venv/bin/datarax-bench" not in command
        assert "--output-dir /root/results/subset" in command
        assert "--data /root/benchmark-data" in command
        assert "'~/results/subset'" not in command

    def test_remote_benchmark_command_repeats_scenarios_flag(self):
        command = vo._build_remote_bench_command(  # noqa: SLF001
            stage="subset",
            repetitions=3,
            scenarios=["CV-1", "NLP-1"],
        )

        assert command.count("--scenarios") == 2
        assert "--scenarios CV-1 --scenarios NLP-1" in command
        assert "--scenarios CV-1 NLP-1" not in command

    def test_remote_benchmark_command_can_disable_wandb(self):
        command = vo._build_remote_bench_command(  # noqa: SLF001
            stage="subset",
            repetitions=3,
            scenarios=["CV-1"],
            use_wandb=False,
        )

        assert "--no-wandb" in command


class TestArtifactCollection:
    def test_collect_artifacts_falls_back_to_scp_when_sky_rsync_missing(
        self,
        tmp_path: Path,
    ):
        run_root = tmp_path / "run"
        logs_dir = tmp_path / "logs"

        with patch.object(
            vo,
            "_run_logged_command",
            side_effect=[
                vo.OrchestrationError("Error: No such command 'rsync'."),
                _command_result(stdout="status ok"),
                _command_result(stdout="scp ok"),
            ],
        ) as run_cmd:
            vo.collect_artifacts(
                cluster="datarax-vast-a100",
                run_root=run_root,
                logs_dir=logs_dir,
                sky_executable="sky",
                live_peek=False,
            )

        assert run_cmd.call_count == 3
        first = run_cmd.call_args_list[0].args[0]
        second = run_cmd.call_args_list[1].args[0]
        third = run_cmd.call_args_list[2].args[0]
        assert first[:2] == ["sky", "rsync"]
        assert second[:2] == ["sky", "status"]
        assert third[0] == "scp"

    def test_collect_artifacts_raises_for_non_rsync_failure(self, tmp_path: Path):
        with patch.object(
            vo,
            "_run_logged_command",
            side_effect=vo.OrchestrationError("permission denied"),
        ):
            with pytest.raises(vo.OrchestrationError, match="permission denied"):
                vo.collect_artifacts(
                    cluster="datarax-vast-a100",
                    run_root=tmp_path / "run",
                    logs_dir=tmp_path / "logs",
                    sky_executable="sky",
                    live_peek=False,
                )

    def test_collect_artifacts_uses_scp_path_when_transfer_mode_is_scp(
        self,
        tmp_path: Path,
    ):
        run_root = tmp_path / "run"
        logs_dir = tmp_path / "logs"

        with patch.object(
            vo,
            "_run_logged_command",
            side_effect=[
                _command_result(stdout="status ok"),
                _command_result(stdout="scp ok"),
            ],
        ) as run_cmd:
            vo.collect_artifacts(
                cluster="datarax-vast-a100",
                run_root=run_root,
                logs_dir=logs_dir,
                sky_executable="sky",
                live_peek=False,
                transfer_method="scp",
            )

        assert run_cmd.call_count == 2
        first = run_cmd.call_args_list[0].args[0]
        second = run_cmd.call_args_list[1].args[0]
        assert first[:2] == ["sky", "status"]
        assert second[0] == "scp"

    def test_collect_artifacts_forced_rsync_raises_with_upgrade_hint(self, tmp_path: Path):
        with patch.object(
            vo,
            "_run_logged_command",
            side_effect=vo.OrchestrationError("Error: No such command 'rsync'."),
        ):
            with pytest.raises(vo.OrchestrationError, match="Suggested upgrade command"):
                vo.collect_artifacts(
                    cluster="datarax-vast-a100",
                    run_root=tmp_path / "run",
                    logs_dir=tmp_path / "logs",
                    sky_executable="sky",
                    live_peek=False,
                    transfer_method="sky-rsync",
                    sky_upgrade_hint='./.venv/bin/pip install -U "skypilot[vast]"',
                )

    def test_collect_artifacts_scp_retries_when_dot_suffix_is_rejected(
        self,
        tmp_path: Path,
    ):
        run_root = tmp_path / "run"
        logs_dir = tmp_path / "logs"

        with patch.object(
            vo,
            "_run_logged_command",
            side_effect=[
                _command_result(stdout="status ok"),
                vo.OrchestrationError("error: unexpected filename: ."),
                _command_result(stdout="scp retry ok"),
            ],
        ) as run_cmd:
            vo.collect_artifacts(
                cluster="datarax-vast-a100",
                run_root=run_root,
                logs_dir=logs_dir,
                sky_executable="sky",
                live_peek=False,
                transfer_method="scp",
            )

        assert run_cmd.call_count == 3
        first_scp = run_cmd.call_args_list[1].args[0]
        second_scp = run_cmd.call_args_list[2].args[0]
        assert first_scp[0] == "scp"
        assert second_scp[0] == "scp"
        assert first_scp[-2] == "datarax-vast-a100:~/results"
        assert second_scp[-2] == "datarax-vast-a100:/root/results"

    def test_collect_artifacts_normalizes_nested_results_layout(self, tmp_path: Path):
        run_root = tmp_path / "run"
        logs_dir = tmp_path / "logs"
        local_target = run_root / "results"

        def _run_cmd(args, *cmd_args, **cmd_kwargs):  # noqa: ANN001
            del cmd_args, cmd_kwargs
            if args and args[0] == "scp":
                nested_subset = local_target / "results" / "subset"
                nested_subset.mkdir(parents=True, exist_ok=True)
                (nested_subset / "manifest.json").write_text("{}")
            return _command_result(stdout="ok")

        with patch.object(vo, "_run_logged_command", side_effect=_run_cmd):
            vo.collect_artifacts(
                cluster="datarax-vast-a100",
                run_root=run_root,
                logs_dir=logs_dir,
                sky_executable="sky",
                live_peek=False,
                transfer_method="scp",
            )

        assert (local_target / "subset" / "manifest.json").exists()
        assert not (local_target / "results").exists()


class TestStageExecution:
    def test_run_stage_requests_gpu_resources_for_exec(self, tmp_path: Path):
        with (
            patch.object(vo, "exec_remote", return_value=_command_result(stdout="ok")) as exec_mock,
            patch.object(vo, "collect_artifacts"),
            patch.object(
                vo,
                "validate_results",
                return_value={"ok": True, "stage": "subset"},
            ),
        ):
            vo.run_stage(
                vo.StageRunConfig(
                    cluster="datarax-vast-a100",
                    run_root=tmp_path / "run",
                    logs_dir=tmp_path / "logs",
                    stage="subset",
                    repetitions=3,
                    scenarios=["CV-1"],
                    sky_executable="sky",
                    live_peek=False,
                    gpu_resource="A100:1",
                ),
            )

        kwargs = exec_mock.call_args.kwargs
        assert kwargs["gpus"] == "A100:1"
        assert kwargs["secret_env_vars"] == ["WANDB_API_KEY"]

    def test_run_stage_disables_wandb_when_requested(self, tmp_path: Path):
        with (
            patch.object(vo, "exec_remote", return_value=_command_result(stdout="ok")) as exec_mock,
            patch.object(vo, "collect_artifacts"),
            patch.object(
                vo,
                "validate_results",
                return_value={"ok": True, "stage": "subset"},
            ),
        ):
            vo.run_stage(
                vo.StageRunConfig(
                    cluster="datarax-vast-a100",
                    run_root=tmp_path / "run",
                    logs_dir=tmp_path / "logs",
                    stage="subset",
                    repetitions=3,
                    scenarios=["CV-1"],
                    sky_executable="sky",
                    live_peek=False,
                    gpu_resource="A100:1",
                    use_wandb=False,
                ),
            )

        command = exec_mock.call_args.kwargs["command"]
        assert "--no-wandb" in command
        assert exec_mock.call_args.kwargs["secret_env_vars"] == []


class TestOrchestrate:
    def test_orchestrate_forwards_launch_compatibility_hints(self, tmp_path: Path):
        args = _args(tmp_path, mode="subset", yes=True)

        with (
            patch.object(
                vo,
                "preflight_checks",
                return_value={
                    "ok": True,
                    "sky_upgrade_hint": "pip install -U 'skypilot[vast]'",
                    "click_pin_hint": "pip install 'click<8.3'",
                },
            ),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster") as launch_cluster,
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(vo, "run_stage", return_value={"ok": True, "stage": "subset"}),
            patch.object(vo, "teardown_cluster"),
        ):
            rc = vo.orchestrate(args)

        assert rc == 0
        assert (
            launch_cluster.call_args.kwargs["sky_upgrade_hint"] == "pip install -U 'skypilot[vast]'"
        )
        assert launch_cluster.call_args.kwargs["click_pin_hint"] == "pip install 'click<8.3'"

    def test_dry_run_skips_launch_and_writes_plan(self, tmp_path: Path):
        args = _args(tmp_path, dry_run=True)
        run_root = Path(args.download_dir) / args.run_id

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster") as launch_cluster,
        ):
            rc = vo.orchestrate(args)

        assert rc == 0
        launch_cluster.assert_not_called()
        summary = json.loads((run_root / "validation_report.json").read_text())
        assert summary["ok"] is True
        assert summary["dry_run"] is True

    def test_launch_failure_falls_back_to_spot(self, tmp_path: Path):
        args = _args(tmp_path, mode="subset", yes=True, allow_spot_fallback=True)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(
                vo,
                "generate_sky_yaml",
                side_effect=[
                    tmp_path / "generated_on_demand.yaml",
                    tmp_path / "generated_spot.yaml",
                ],
            ) as generate_sky_yaml,
            patch.object(
                vo,
                "launch_cluster",
                side_effect=[vo.OrchestrationError("on-demand failed"), None],
            ) as launch_cluster,
            patch.object(vo, "_is_cluster_visible", return_value=False),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(vo, "run_stage", return_value={"ok": True, "stage": "subset"}),
            patch.object(vo, "teardown_cluster"),
        ):
            rc = vo.orchestrate(args)

        assert rc == 0
        assert launch_cluster.call_count == 2
        assert generate_sky_yaml.call_count == 2
        assert generate_sky_yaml.call_args_list[1].kwargs["on_demand"] is False

    def test_orchestrate_disables_remote_wandb_when_api_key_missing(self, tmp_path: Path):
        args = _args(tmp_path, mode="subset", yes=True)

        with (
            patch.object(
                vo,
                "preflight_checks",
                return_value={"ok": True, "wandb_api_key_set": False},
            ),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster"),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(
                vo, "run_stage", return_value={"ok": True, "stage": "subset"}
            ) as run_stage,
            patch.object(vo, "teardown_cluster"),
        ):
            rc = vo.orchestrate(args)

        assert rc == 0
        assert run_stage.call_args.args[0].use_wandb is False

    def test_timeout_retries_same_cluster_before_fallback(self, tmp_path: Path):
        args = _args(tmp_path, mode="subset", yes=True, allow_spot_fallback=True)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml") as gen,
            patch.object(
                vo,
                "launch_cluster",
                side_effect=[vo.OrchestrationError("Command timed out after 300s"), None],
            ) as launch_cluster,
            patch.object(vo, "_is_cluster_visible", return_value=True),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(vo, "run_stage", return_value={"ok": True, "stage": "subset"}),
            patch.object(vo, "teardown_cluster"),
        ):
            rc = vo.orchestrate(args)

        assert rc == 0
        assert launch_cluster.call_count == 2
        assert gen.call_count == 1
        assert launch_cluster.call_args_list[1].kwargs["log_name"] == "sky_launch_retry.log"

    def test_stall_failure_does_not_trigger_spot_fallback(self, tmp_path: Path):
        args = _args(tmp_path, mode="subset", yes=True, allow_spot_fallback=True)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(
                vo,
                "generate_sky_yaml",
                side_effect=[
                    tmp_path / "generated_on_demand.yaml",
                    tmp_path / "generated_spot.yaml",
                ],
            ) as gen,
            patch.object(
                vo,
                "launch_cluster",
                side_effect=vo.OrchestrationError(
                    "Stall detected: command produced no output for 300s",
                ),
            ) as launch_cluster,
            patch.object(vo, "teardown_cluster") as teardown,
        ):
            with pytest.raises(vo.OrchestrationError, match="Stall detected"):
                vo.orchestrate(args)

        assert launch_cluster.call_count == 1
        assert gen.call_count == 1
        teardown.assert_called_once()

    def test_emits_progress_messages(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        args = _args(tmp_path, analyze=False)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster"),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(
                vo,
                "run_stage",
                side_effect=[
                    {"ok": True, "stage": "subset"},
                    {"ok": True, "stage": "full"},
                ],
            ),
            patch.object(vo, "teardown_cluster"),
        ):
            rc = vo.orchestrate(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "Starting run_id=test_run" in out
        assert "Launching cluster (this may take several minutes)" in out
        assert "Running subset stage" in out
        assert "Running full stage" in out
        assert "Run completed successfully" in out

    def test_launch_failure_still_attempts_teardown(self, tmp_path: Path):
        args = _args(tmp_path, analyze=False)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(
                vo,
                "launch_cluster",
                side_effect=vo.OrchestrationError("launch failed"),
            ),
            patch.object(vo, "teardown_cluster") as teardown,
        ):
            with pytest.raises(vo.OrchestrationError, match="launch failed"):
                vo.orchestrate(args)

        teardown.assert_called_once()

    def test_launch_failure_writes_failure_report(self, tmp_path: Path):
        args = _args(tmp_path, analyze=False)
        run_root = Path(args.download_dir) / args.run_id

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(
                vo,
                "launch_cluster",
                side_effect=vo.OrchestrationError("launch failed"),
            ),
            patch.object(vo, "teardown_cluster"),
        ):
            with pytest.raises(vo.OrchestrationError, match="launch failed"):
                vo.orchestrate(args)

        summary = json.loads((run_root / "validation_report.json").read_text())
        assert summary["ok"] is False
        assert "launch failed" in summary["error"]

    def test_two_pass_happy_path(self, tmp_path: Path):
        args = _args(tmp_path, analyze=True)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster"),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(
                vo,
                "run_stage",
                side_effect=[
                    {"ok": True, "stage": "subset"},
                    {"ok": True, "stage": "full"},
                ],
            ) as run_stage,
            patch.object(vo, "run_analysis") as run_analysis,
            patch.object(vo, "teardown_cluster") as teardown,
        ):
            rc = vo.orchestrate(args)

        assert rc == 0
        assert run_stage.call_count == 2
        run_analysis.assert_called_once()
        teardown.assert_called_once()

    def test_hardware_mismatch_fails_fast(self, tmp_path: Path):
        args = _args(tmp_path, analyze=False)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster"),
            patch.object(
                vo,
                "verify_remote_hardware",
                side_effect=vo.OrchestrationError("bad hardware"),
            ),
            patch.object(vo, "run_stage") as run_stage,
            patch.object(vo, "teardown_cluster") as teardown,
        ):
            with pytest.raises(vo.OrchestrationError, match="bad hardware"):
                vo.orchestrate(args)

        run_stage.assert_not_called()
        teardown.assert_called_once()

    def test_subset_validation_failure_skips_full(self, tmp_path: Path):
        args = _args(tmp_path, mode="two-pass")

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster"),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(
                vo,
                "run_stage",
                return_value={"ok": False, "stage": "subset"},
            ) as run_stage,
            patch.object(vo, "teardown_cluster") as teardown,
        ):
            with pytest.raises(vo.OrchestrationError, match="Subset validation failed"):
                vo.orchestrate(args)

        assert run_stage.call_count == 1
        teardown.assert_called_once()

    def test_analysis_failure_still_tears_down(self, tmp_path: Path):
        args = _args(tmp_path, mode="subset", analyze=True)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "launch_cluster"),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(vo, "run_stage", return_value={"ok": True, "stage": "subset"}),
            patch.object(
                vo,
                "run_analysis",
                side_effect=vo.OrchestrationError("analysis failed"),
            ),
            patch.object(vo, "teardown_cluster") as teardown,
        ):
            with pytest.raises(vo.OrchestrationError, match="analysis failed"):
                vo.orchestrate(args)

        teardown.assert_called_once()

    def test_teardown_failure_does_not_mask_primary_stage_failure(self, tmp_path: Path):
        args = _args(tmp_path, mode="subset", yes=True)

        with (
            patch.object(vo, "preflight_checks", return_value={"ok": True}),
            patch.object(vo, "generate_sky_yaml", return_value=tmp_path / "generated.yaml"),
            patch.object(vo, "_confirm", return_value=True),
            patch.object(vo, "launch_cluster"),
            patch.object(vo, "verify_remote_hardware", return_value={"ok": True}),
            patch.object(vo, "run_stage", side_effect=vo.OrchestrationError("subset failed")),
            patch.object(
                vo,
                "teardown_cluster",
                side_effect=vo.OrchestrationError("teardown failed"),
            ),
        ):
            with pytest.raises(vo.OrchestrationError, match="subset failed"):
                vo.orchestrate(args)


class TestCommandExecution:
    def test_exec_remote_forwards_secret_env_vars(self, tmp_path: Path):
        with patch.object(vo, "_run_logged_command", return_value=_command_result()) as run_cmd:
            vo.exec_remote(
                cluster="datarax-vast-a100",
                command="echo hello",
                logs_dir=tmp_path,
                log_name="exec.log",
                sky_executable="sky",
                live_peek=False,
                gpus="A100:1",
                secret_env_vars=["WANDB_API_KEY", "HF_TOKEN"],
            )

        args = run_cmd.call_args.args[0]
        assert args[:3] == ["sky", "exec", "datarax-vast-a100"]
        assert "--gpus" in args
        assert "--secret" in args
        secret_values = [
            args[idx + 1] for idx, token in enumerate(args[:-1]) if token == "--secret"
        ]
        assert secret_values == ["HF_TOKEN", "WANDB_API_KEY"]

    def test_status_wraps_on_narrow_terminals(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            vo_utils.shutil,
            "get_terminal_size",
            lambda fallback: (fallback, os.terminal_size((60, 24)))[1],
        )

        vo._status(  # noqa: SLF001
            "Running command: "
            + "/a/very/long/path/that/should/be/split/for/smaller/terminals/"
            + "and/not/force/native-terminal-wrapping",
        )

        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) >= 2
        assert all("[INFO]" in line for line in lines)
        assert all(len(line) <= 60 for line in lines)

    def test_status_transient_updates_single_terminal_line(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(vo_utils.sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(vo_utils, "_TRANSIENT_ACTIVE", False)

        vo._status("working...", transient=True)  # noqa: SLF001
        vo._status("done")  # noqa: SLF001

        out = capsys.readouterr().out
        assert "\r" in out
        assert "working..." in out
        assert "done" in out

    def test_status_logs_scroll_above_transient_progress(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(vo_utils.sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(vo_utils, "_TRANSIENT_ACTIVE", False)
        monkeypatch.setattr(vo_utils, "_TRANSIENT_LINE", "")
        monkeypatch.setattr(
            vo_utils.shutil,
            "get_terminal_size",
            lambda fallback: (fallback, os.terminal_size((100, 24)))[1],
        )

        vo._status("progress-line", transient=True)  # noqa: SLF001
        vo._status("log line one")  # noqa: SLF001
        vo._status("log line two")  # noqa: SLF001

        out = capsys.readouterr().out
        assert "log line one" in out
        assert "log line two" in out
        assert out.rfind("progress-line") > out.rfind("log line two")

    def test_missing_executable_is_wrapped(self, tmp_path: Path):
        log_path = tmp_path / "missing_cmd.log"

        with pytest.raises(vo.OrchestrationError, match="Failed to execute command"):
            vo._run_logged_command(  # noqa: SLF001
                ["/definitely/missing-executable-for-datarax-tests"],
                log_path,
            )

        assert log_path.exists()
        text = log_path.read_text()
        assert "failed_to_start" in text

    def test_timeout_is_wrapped(self, tmp_path: Path):
        log_path = tmp_path / "timeout_cmd.log"

        with pytest.raises(vo.OrchestrationError, match="timed out"):
            vo._run_logged_command(  # noqa: SLF001
                ["bash", "-lc", "sleep 2"],
                log_path,
                timeout_sec=1,
            )

        text = log_path.read_text()
        assert "timed_out:" in text

    def test_run_logged_command_clears_transient_progress_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        log_path = tmp_path / "transient_cleanup.log"
        monkeypatch.setattr(vo_utils.sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(vo_utils, "_TRANSIENT_ACTIVE", False)
        monkeypatch.setattr(vo_utils, "_TRANSIENT_LINE", "")

        vo._run_logged_command(  # noqa: SLF001
            ["bash", "-lc", "sleep 2"],
            log_path,
            live_peek=False,
            _heartbeat_interval_sec=1,
        )

        assert vo_utils._TRANSIENT_ACTIVE is False  # noqa: SLF001
        assert vo_utils._TRANSIENT_LINE == ""  # noqa: SLF001

    def test_run_logged_command_caps_capture_but_keeps_full_log(self, tmp_path: Path):
        log_path = tmp_path / "capture_cap.log"
        result = vo._run_logged_command(  # noqa: SLF001
            [
                "bash",
                "-lc",
                "for i in $(seq 1 200); do echo line-$i; done",
            ],
            log_path,
            live_peek=False,
            capture_limit_chars=256,
        )

        assert "truncated" in result.stdout.lower()
        assert "line-200" in result.stdout
        assert "\nline-1\n" not in f"\n{result.stdout}"

        full_log = log_path.read_text()
        assert "line-1" in full_log
        assert "line-200" in full_log

    def test_live_peek_emits_intermediate_command_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        log_path = tmp_path / "peek_cmd.log"

        vo._run_logged_command(  # noqa: SLF001
            [
                "bash",
                "-lc",
                "echo setup-start; sleep 2; echo benchmark-running; sleep 2; echo finished",
            ],
            log_path,
            live_peek=True,
            peek_interval_sec=1,
        )

        out = capsys.readouterr().out
        assert "peek:" in out
        assert "setup-start" in out or "benchmark-running" in out

    def test_live_peek_deduplicates_unchanged_lines(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        log_path = tmp_path / "peek_dedup_cmd.log"

        vo._run_logged_command(  # noqa: SLF001
            [
                "bash",
                "-lc",
                "echo same-line; sleep 3; echo done-line",
            ],
            log_path,
            live_peek=True,
            peek_interval_sec=1,
        )

        out = capsys.readouterr().out
        # The same progress line should not be emitted repeatedly every interval.
        assert out.count("peek: same-line") <= 1

    def test_idle_output_notice_emits_when_command_goes_silent(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        log_path = tmp_path / "peek_idle_cmd.log"

        vo._run_logged_command(  # noqa: SLF001
            [
                "bash",
                "-lc",
                "echo warmup-line; sleep 3; echo done-line",
            ],
            log_path,
            live_peek=True,
            peek_interval_sec=1,
            _heartbeat_interval_sec=1,
            _idle_notice_sec=2,
        )

        out = capsys.readouterr().out
        assert "no new command output for" in out

    def test_phase_notice_after_setup_install_complete(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        log_path = tmp_path / "peek_phase_notice_cmd.log"

        vo._run_logged_command(  # noqa: SLF001
            [
                "bash",
                "-lc",
                "echo '(setup pid=1) Installed 226 packages in 23.74s'; sleep 2; echo done-line",
            ],
            log_path,
            live_peek=True,
            peek_interval_sec=1,
            _heartbeat_interval_sec=1,
            _idle_notice_sec=2,
        )

        out = capsys.readouterr().out
        assert "Remote dependency install appears complete" in out

    def test_idle_output_notice_includes_phase_hint(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        log_path = tmp_path / "peek_phase_idle_cmd.log"

        vo._run_logged_command(  # noqa: SLF001
            [
                "bash",
                "-lc",
                "echo '(setup pid=1) Installed 226 packages in 23.74s'; sleep 3; echo done-line",
            ],
            log_path,
            live_peek=True,
            peek_interval_sec=1,
            _heartbeat_interval_sec=1,
            _idle_notice_sec=2,
        )

        out = capsys.readouterr().out
        assert "phase:" in out
        assert "setup_complete" in out

    def test_idle_notice_explains_post_setup_pipeline_state(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        log_path = tmp_path / "peek_setup_transition_cmd.log"

        vo._run_logged_command(  # noqa: SLF001
            [
                "bash",
                "-lc",
                "echo '(setup pid=1) Installed 226 packages in 23.74s'; sleep 8; echo done-line",
            ],
            log_path,
            live_peek=True,
            peek_interval_sec=1,
            _heartbeat_interval_sec=1,
            _idle_notice_sec=3,
        )

        out = capsys.readouterr().out
        assert "Dependency installation finished; benchmark pipeline has moved past setup" in out

    def test_stall_detector_runs_diagnostics_and_fails_fast(
        self,
        tmp_path: Path,
    ):
        log_path = tmp_path / "stall_detector_cmd.log"
        calls: list[bool] = []

        def _diagnostics() -> dict[str, str]:
            calls.append(True)
            return {
                "queue": "queue diagnostics output",
                "logs_tail": "logs diagnostics output",
            }

        with pytest.raises(vo.OrchestrationError, match="Stall detected"):
            vo._run_logged_command(  # noqa: SLF001
                [
                    "bash",
                    "-lc",
                    "echo begin-line; sleep 5; echo done-line",
                ],
                log_path,
                live_peek=True,
                peek_interval_sec=1,
                _heartbeat_interval_sec=1,
                _idle_notice_sec=10,
                stall_timeout_sec=2,
                stall_diagnostics=_diagnostics,
            )

        assert calls == [True]
        text = log_path.read_text()
        assert "stalled: no output for" in text
        assert "queue diagnostics output" in text
        assert "logs diagnostics output" in text


class TestStallDiagnostics:
    def test_run_stall_diagnostics_collects_queue_and_logs(self, tmp_path: Path):
        with patch.object(
            vo,
            "_run_logged_command",
            side_effect=[
                _command_result(stdout="queue ok", code=0),
                _command_result(stdout="logs ok", code=0),
            ],
        ) as run_cmd:
            report = vo._run_stall_diagnostics(  # noqa: SLF001
                cluster="datarax-vast-a100",
                logs_dir=tmp_path,
                sky_executable="sky",
            )

        assert "queue ok" in report["queue"]
        assert "logs ok" in report["logs_tail"]
        calls = [c.args[0] for c in run_cmd.call_args_list]
        assert any(cmd[1] == "queue" for cmd in calls)
        assert any(cmd[1] == "logs" for cmd in calls)


class TestLaunchCommand:
    def test_launch_cluster_uses_non_interactive_yes(self, tmp_path: Path):
        with patch.object(vo, "_run_logged_command") as run_cmd:
            vo.launch_cluster(
                yaml_path=tmp_path / "cfg.yaml",
                infra="vast",
                cluster_name="datarax-vast-a100",
                logs_dir=tmp_path,
                sky_executable="sky",
                launch_timeout_sec=300,
                log_name="launch.log",
            )

        run_args = run_cmd.call_args.args[0]
        assert run_args[:4] == ["sky", "launch", str(tmp_path / "cfg.yaml"), "--infra"]
        assert "-c" in run_args
        assert "datarax-vast-a100" in run_args
        assert "-y" in run_args

    def test_teardown_cluster_uses_non_interactive_yes(self, tmp_path: Path):
        with patch.object(vo, "_run_logged_command") as run_cmd:
            vo.teardown_cluster(
                cluster="datarax-vast-a100",
                logs_dir=tmp_path,
                sky_executable="sky",
            )

        run_args = run_cmd.call_args.args[0]
        assert run_args[:2] == ["sky", "down"]
        assert "datarax-vast-a100" in run_args
        assert "-y" in run_args

    def test_launch_cluster_forwards_capture_limit_to_command_runner(self, tmp_path: Path):
        with patch.object(vo, "_run_logged_command") as run_cmd:
            vo.launch_cluster(
                yaml_path=tmp_path / "cfg.yaml",
                infra="vast",
                cluster_name="datarax-vast-a100",
                logs_dir=tmp_path,
                sky_executable="sky",
                launch_timeout_sec=300,
                capture_limit_chars=4096,
            )

        assert run_cmd.call_args.kwargs["capture_limit_chars"] == 4096

    def test_launch_cluster_rewraps_backend_false_with_fix_hints(self, tmp_path: Path):
        with patch.object(
            vo,
            "_run_logged_command",
            side_effect=vo.OrchestrationError("ValueError: False backend is not supported."),
        ):
            with pytest.raises(vo.OrchestrationError, match="backend parsing incompatibility"):
                vo.launch_cluster(
                    yaml_path=tmp_path / "cfg.yaml",
                    infra="vast",
                    cluster_name="datarax-vast-a100",
                    logs_dir=tmp_path,
                    sky_executable="sky",
                    launch_timeout_sec=300,
                    sky_upgrade_hint="pip install -U 'skypilot[vast]'",
                    click_pin_hint="pip install 'click<8.3'",
                )


class TestParser:
    def test_allowed_gpu_defaults_to_a100_when_not_set(self):
        parser = vo.build_parser()
        args = parser.parse_args([])

        tokens = vo._effective_allowed_gpu_tokens(args.allowed_gpu)  # noqa: SLF001
        assert tokens == ["A100"]

    def test_allowed_gpu_override_does_not_include_default(self):
        parser = vo.build_parser()
        args = parser.parse_args(["--allowed-gpu", "H100"])

        tokens = vo._effective_allowed_gpu_tokens(args.allowed_gpu)  # noqa: SLF001
        assert tokens == ["H100"]

    def test_capture_limit_parser_and_normalization(self):
        parser = vo.build_parser()
        default_args = parser.parse_args([])
        assert default_args.capture_limit_chars == 1_000_000
        assert vo._normalize_capture_limit_chars(default_args.capture_limit_chars) == 1_000_000  # noqa: SLF001

        disabled_args = parser.parse_args(["--capture-limit-chars", "0"])
        assert vo._normalize_capture_limit_chars(disabled_args.capture_limit_chars) is None  # noqa: SLF001
