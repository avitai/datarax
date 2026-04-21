"""Profiling infrastructure for trace-backed performance validation.

Captures XProf/Perfetto traces for benchmark scenarios, storing them
alongside commit hash and XLA flag configuration. Each P0 optimization
change should produce before/after traces for validation.

Protocol ref: Section 8.2 of the performance audit.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceArtifact:
    """Metadata for a captured profiling trace.

    Attributes:
        scenario_id: Scenario that was profiled.
        variant_name: Variant within the scenario.
        framework: Framework name (e.g., "Datarax").
        backend: JAX backend used (cpu/gpu/tpu).
        commit_hash: Git commit hash at capture time.
        xla_flags: XLA_FLAGS environment variable value.
        trace_dir: Directory containing the raw trace files.
        duration_sec: Duration of the profiled section.
        timestamp: Unix timestamp of capture.
    """

    scenario_id: str
    variant_name: str
    framework: str
    backend: str
    commit_hash: str
    xla_flags: str
    trace_dir: Path
    duration_sec: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "scenario_id": self.scenario_id,
            "variant_name": self.variant_name,
            "framework": self.framework,
            "backend": self.backend,
            "commit_hash": self.commit_hash,
            "xla_flags": self.xla_flags,
            "trace_dir": str(self.trace_dir),
            "duration_sec": self.duration_sec,
            "timestamp": self.timestamp,
        }


def get_git_commit_hash() -> str:
    """Get the current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(  # nosec B603, B607
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def get_xla_flags() -> str:
    """Get the current XLA_FLAGS environment variable value."""
    return os.environ.get("XLA_FLAGS", "")


def capture_jax_trace(
    output_dir: Path,
    scenario_id: str,
    variant_name: str,
    framework: str,
    run_fn: Any,
    duration_ms: int = 2000,
) -> TraceArtifact | None:
    """Capture a JAX profiling trace during scenario execution.

    Uses jax.profiler.trace() to capture XLA execution traces. The trace
    is saved to output_dir with metadata for later analysis.

    Args:
        output_dir: Directory to store trace artifacts.
        scenario_id: Scenario being profiled.
        variant_name: Variant name.
        framework: Framework name.
        run_fn: Callable that executes the workload to profile.
            Called as ``run_fn()`` within the trace context.
        duration_ms: Trace duration hint in milliseconds.

    Returns:
        TraceArtifact with metadata, or None if profiling unavailable.
    """
    del duration_ms
    try:
        import jax
    except ImportError:
        _logger.warning("JAX not available; skipping trace capture")
        return None

    backend = jax.default_backend()
    commit = get_git_commit_hash()
    xla_flags = get_xla_flags()

    trace_dir = output_dir / f"trace_{scenario_id}_{variant_name}_{commit}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        with jax.profiler.trace(str(trace_dir)):
            run_fn()
    except (RuntimeError, OSError, ValueError) as exc:
        _logger.warning("Trace capture failed: %s", exc)
        return None
    duration = time.perf_counter() - start

    artifact = TraceArtifact(
        scenario_id=scenario_id,
        variant_name=variant_name,
        framework=framework,
        backend=backend,
        commit_hash=commit,
        xla_flags=xla_flags,
        trace_dir=trace_dir,
        duration_sec=duration,
    )

    # Save metadata alongside trace files
    metadata_path = trace_dir / "trace_metadata.json"
    metadata_path.write_text(json.dumps(artifact.to_dict(), indent=2))
    _logger.info(
        "Trace captured: %s/%s -> %s (%.2fs)",
        scenario_id,
        variant_name,
        trace_dir,
        duration,
    )

    return artifact


class TraceStore:
    """Manages a collection of trace artifacts for comparison.

    Stores traces organized by scenario/variant/commit for before/after
    comparison of optimization changes.
    """

    def __init__(self, base_dir: Path | str) -> None:
        """Initialize the store with a base directory for trace artifacts."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_traces(self, scenario_id: str | None = None) -> list[TraceArtifact]:
        """List all stored trace artifacts, optionally filtered by scenario."""
        artifacts: list[TraceArtifact] = []
        for metadata_path in self.base_dir.rglob("trace_metadata.json"):
            try:
                data = json.loads(metadata_path.read_text())
                if scenario_id and data.get("scenario_id") != scenario_id:
                    continue
                artifacts.append(
                    TraceArtifact(
                        scenario_id=data["scenario_id"],
                        variant_name=data["variant_name"],
                        framework=data["framework"],
                        backend=data["backend"],
                        commit_hash=data["commit_hash"],
                        xla_flags=data["xla_flags"],
                        trace_dir=Path(data["trace_dir"]),
                        duration_sec=data["duration_sec"],
                        timestamp=data.get("timestamp", 0.0),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                _logger.warning("Skipping invalid trace metadata %s: %s", metadata_path, exc)
        return sorted(artifacts, key=lambda a: a.timestamp)

    def get_latest(self, scenario_id: str, variant_name: str) -> TraceArtifact | None:
        """Get the most recent trace for a scenario/variant."""
        traces = [t for t in self.list_traces(scenario_id) if t.variant_name == variant_name]
        return traces[-1] if traces else None
