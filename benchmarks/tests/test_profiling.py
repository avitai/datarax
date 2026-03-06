"""Tests for the profiling infrastructure module."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.core.profiling import (
    capture_jax_trace,
    get_git_commit_hash,
    get_xla_flags,
    TraceArtifact,
    TraceStore,
)


class TestGetGitCommitHash:
    """Test git commit hash retrieval."""

    def test_returns_string(self) -> None:
        result = get_git_commit_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_short_hash(self) -> None:
        result = get_git_commit_hash()
        # Short hash is typically 7-12 chars, or "unknown"
        assert len(result) <= 12 or result == "unknown"


class TestGetXlaFlags:
    """Test XLA flags retrieval."""

    def test_returns_string(self) -> None:
        result = get_xla_flags()
        assert isinstance(result, str)


class TestTraceArtifact:
    """Test TraceArtifact dataclass."""

    def test_to_dict(self, tmp_path: Path) -> None:
        artifact = TraceArtifact(
            scenario_id="CV-1",
            variant_name="small",
            framework="Datarax",
            backend="cpu",
            commit_hash="abc1234",
            xla_flags="--xla_cpu_enable_fast_math=true",
            trace_dir=tmp_path / "traces",
            duration_sec=1.5,
            timestamp=1234567890.0,
        )
        d = artifact.to_dict()
        assert d["scenario_id"] == "CV-1"
        assert d["framework"] == "Datarax"
        assert d["commit_hash"] == "abc1234"
        assert d["duration_sec"] == 1.5


class TestCaptureJaxTrace:
    """Test JAX trace capture."""

    def test_captures_trace_with_simple_workload(self, tmp_path: Path) -> None:
        import jax.numpy as jnp

        def workload() -> None:
            x = jnp.ones((100, 100))
            _ = (x @ x).block_until_ready()

        artifact = capture_jax_trace(
            output_dir=tmp_path,
            scenario_id="CV-1",
            variant_name="test",
            framework="Datarax",
            run_fn=workload,
        )

        assert artifact is not None
        assert artifact.scenario_id == "CV-1"
        assert artifact.framework == "Datarax"
        assert artifact.duration_sec > 0
        assert artifact.trace_dir.exists()

        # Check metadata was written
        metadata_path = artifact.trace_dir / "trace_metadata.json"
        assert metadata_path.exists()
        data = json.loads(metadata_path.read_text())
        assert data["scenario_id"] == "CV-1"


class TestTraceStore:
    """Test trace artifact storage and retrieval."""

    def test_list_empty(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "traces")
        assert store.list_traces() == []

    def test_list_after_capture(self, tmp_path: Path) -> None:
        import jax.numpy as jnp

        store = TraceStore(tmp_path / "traces")

        artifact = capture_jax_trace(
            output_dir=store.base_dir,
            scenario_id="CV-1",
            variant_name="small",
            framework="Datarax",
            run_fn=lambda: jnp.ones(10).block_until_ready(),
        )
        assert artifact is not None

        traces = store.list_traces()
        assert len(traces) == 1
        assert traces[0].scenario_id == "CV-1"

    def test_filter_by_scenario(self, tmp_path: Path) -> None:
        import jax.numpy as jnp

        store = TraceStore(tmp_path / "traces")

        # Capture two different scenarios
        capture_jax_trace(
            output_dir=store.base_dir,
            scenario_id="CV-1",
            variant_name="small",
            framework="Datarax",
            run_fn=lambda: jnp.ones(10).block_until_ready(),
        )
        capture_jax_trace(
            output_dir=store.base_dir,
            scenario_id="NLP-1",
            variant_name="small",
            framework="Datarax",
            run_fn=lambda: jnp.ones(10).block_until_ready(),
        )

        cv_traces = store.list_traces(scenario_id="CV-1")
        assert len(cv_traces) == 1
        assert cv_traces[0].scenario_id == "CV-1"

    def test_get_latest(self, tmp_path: Path) -> None:
        import jax.numpy as jnp

        store = TraceStore(tmp_path / "traces")

        capture_jax_trace(
            output_dir=store.base_dir,
            scenario_id="CV-1",
            variant_name="small",
            framework="Datarax",
            run_fn=lambda: jnp.ones(10).block_until_ready(),
        )

        latest = store.get_latest("CV-1", "small")
        assert latest is not None
        assert latest.scenario_id == "CV-1"

    def test_get_latest_returns_none_for_missing(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / "traces")
        assert store.get_latest("MISSING", "small") is None
