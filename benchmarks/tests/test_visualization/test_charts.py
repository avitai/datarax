"""Tests for benchmark chart generation.

RED phase: defines expected behavior for ChartGenerator's 7 chart types.
"""

from __future__ import annotations

from pathlib import Path


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
