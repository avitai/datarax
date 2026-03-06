"""Energon adapter for the benchmark framework.

Wraps NVIDIA's Megatron Energon multi-modal data loader with the
PipelineAdapter lifecycle. Tier 3 -- only supports MM-1.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters._utils import cleanup_temp_dir, setup_temp_dir, write_numpy_tar
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


@register
class EnergonAdapter(PipelineAdapter):
    """PipelineAdapter for NVIDIA Megatron Energon.

    Energon is Megatron-LM's multi-modal data loading library. Requires
    separate installation (megatron-energon). Only supports MM-1.
    """

    def __init__(self) -> None:
        """Initialize the Energon adapter."""
        super().__init__()
        self._dataset: Any = None
        self._tmp_dir: Any = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "Energon"

    @property
    def version(self) -> str:
        """Return the Energon version string."""
        import megatron.energon

        return getattr(megatron.energon, "__version__", "unknown")

    def is_available(self) -> bool:
        """Return True if Megatron Energon is installed."""
        try:
            import megatron.energon  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        """Return the set of supported benchmark scenario IDs."""
        return {"MM-1"}

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the Energon pipeline for the given scenario configuration."""
        from megatron.energon import get_train_dataset, WorkerConfig

        data_path, self._tmp_dir = setup_temp_dir(config, "energon_data")

        tar_path = data_path / "shard-000000.tar"
        write_numpy_tar(data, tar_path)

        self._dataset = get_train_dataset(
            path=str(data_path),
            worker_config=WorkerConfig(num_workers=config.num_workers),
            batch_size=config.batch_size,
        )
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._dataset

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        if isinstance(batch, dict):
            return [np.asarray(v) for v in batch.values()]
        return [np.asarray(batch)]

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._dataset = None
        cleanup_temp_dir(self._tmp_dir)
        self._tmp_dir = None
        super().teardown()
