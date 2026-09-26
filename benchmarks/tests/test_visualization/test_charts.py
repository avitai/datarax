"""Tests for benchmark chart generation.

RED phase: defines expected behavior for ChartGenerator's 7 chart types.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
from calibrax.core import BenchmarkResult
from matplotlib.patches import Rectangle


class TestChartGenerator:
    """Tests for the 7 benchmark visualization chart types."""

    def test_throughput_bars_generates_file(self, mock_results, tmp_path: Path):
        """Throughput bar chart must generate a file."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        fig = gen.throughput_bars()
        assert fig is not None

    def test_throughput_radar_generates_file(self, mock_results, tmp_path: Path):
        """Throughput radar chart must generate a file."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        fig = gen.throughput_radar()
        assert fig is not None

    def test_latency_cdf_generates_file(self, mock_results, tmp_path: Path):
        """Latency CDF chart must generate a file."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        fig = gen.latency_cdf(scenario_id="CV-1")
        assert fig is not None

    def test_memory_waterfall_generates_file(self, mock_results, tmp_path: Path):
        """Memory waterfall chart must generate a file."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        fig = gen.memory_waterfall()
        assert fig is not None

    def test_scaling_curves_generates_file(
        self,
        mock_results_with_scaling,
        tmp_path: Path,
    ):
        """Scaling curves chart must generate a file."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results_with_scaling, tmp_path)
        fig = gen.scaling_curves()
        assert fig is not None

    def test_chain_depth_generates_file(self, mock_results, tmp_path: Path):
        """Chain depth degradation chart must generate a file."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        fig = gen.chain_depth()
        assert fig is not None

    def test_chain_depth_reads_the_depth_the_config_recorded(self, tmp_path: Path):
        """The chart plots the depth a PC scenario stored, not one parsed from its id.

        ``ScenarioConfig.extra`` holds the depth, and a record's free-form fields come back
        typed, so the value is read rather than subscripted out of a union.
        """
        from benchmarks.runners.full_runner import ComparativeResults
        from benchmarks.tests.test_analysis.conftest import make_result
        from benchmarks.visualization.charts import ChartGenerator

        results = ComparativeResults(
            results={
                "Datarax": [
                    make_result(
                        scenario_id="PC-1", throughput=throughput, extra={"chain_depth": depth}
                    )
                    for depth, throughput in ((1, 900.0), (3, 600.0), (6, 300.0))
                ]
            },
            environment={"platform": {"backend": "cpu", "device_count": 1}},
            platform="cpu",
            timestamp=time.time(),
        )

        figure = ChartGenerator(results, tmp_path).chain_depth()

        depths = np.asarray(figure.axes[0].lines[0].get_xdata())
        assert [int(depth) for depth in depths] == [1, 3, 6]

    def test_memory_waterfall_estimates_from_the_recorded_size_and_shape(self, tmp_path: Path):
        """Without resource data the bar is the recorded records times shape, 4 bytes each.

        A size recorded as ``None`` was not measured, and counts as the default of none.
        """
        from benchmarks.runners.full_runner import ComparativeResults
        from benchmarks.tests.test_analysis.conftest import make_result
        from benchmarks.visualization.charts import ChartGenerator

        def recorded(framework: str, size: int | None) -> BenchmarkResult:
            result = make_result(framework=framework)
            config = {**result.config, "dataset_size": size, "element_shape": [256, 1024]}
            return dataclasses.replace(result, config=config)

        results = ComparativeResults(
            results={"Datarax": [recorded("Datarax", 8)], "Grain": [recorded("Grain", None)]},
            environment={"platform": {"backend": "cpu", "device_count": 1}},
            platform="cpu",
            timestamp=time.time(),
        )

        figure = ChartGenerator(results, tmp_path).memory_waterfall()

        bars = [patch for patch in figure.axes[0].patches if isinstance(patch, Rectangle)]
        assert [bar.get_height() for bar in bars] == [8.0, 0.0]

    def test_feature_heatmap_generates_file(self, mock_results, tmp_path: Path):
        """Feature heatmap chart must generate a file."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        fig = gen.feature_heatmap()
        assert fig is not None

    def test_generate_all_creates_7_charts(self, mock_results, tmp_path: Path):
        """generate_all must create 7 chart files."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        paths = gen.generate_all(formats=("png",))

        assert len(paths) == 7
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_output_formats_png_and_svg(self, mock_results, tmp_path: Path):
        """generate_all with both formats must produce PNG and SVG."""
        from benchmarks.visualization.charts import ChartGenerator

        gen = ChartGenerator(mock_results, tmp_path)
        paths = gen.generate_all(formats=("png", "svg"))

        # 7 charts × 2 formats = 14 files
        assert len(paths) == 14
        png_count = sum(1 for p in paths if p.suffix == ".png")
        svg_count = sum(1 for p in paths if p.suffix == ".svg")
        assert png_count == 7
        assert svg_count == 7
