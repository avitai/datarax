#!/usr/bin/env python
"""Test Results Analyzer - Test Suite Analysis Dashboard

PURPOSE:
--------
This script provides a detailed analysis dashboard for pytest test results,
using the same successful DataTables-based interface as the skipped tests tracker.
It parses pytest-json-report output to provide detailed failure analysis, performance
metrics, and test suite health visualization.

FEATURES:
---------
1. Test Results Analysis:
   - Parse pytest-json-report output
   - Categorize failures by error type
   - Extract error messages and stack traces
   - Track test performance metrics

2. Interactive Dashboard:
   - DataTables with SearchPanes for filtering
   - Export capabilities (CSV, Excel, PDF)
   - Sortable and searchable columns
   - GitHub integration for viewing tests

3. Multiple Views:
   - Failures & Errors with detailed tracebacks
   - Performance metrics and slow tests
   - Module-level statistics
   - All tests overview

USAGE:
------
1. Generate test results with pytest-json-report:
   ```bash
   uv run pytest -vv --json-report --json-report-file=temp/test-results.json
   ```

2. Generate dashboard:
   ```bash
   python tests/scripts/test_results_analyzer.py
   ```

AUTHOR: Datarax Development Team
VERSION: 2.0.0
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Any
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Datarax Test Results Analysis Dashboard</title>

    <!-- jQuery (required for DataTables) -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>

    <!-- Bootstrap 5 CSS & JS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
          rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js">
    </script>

    <!-- Font Awesome for icons -->
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <!-- DataTables with Bootstrap 5 styling -->
    <link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/searchpanes/2.2.0/css/searchPanes.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/select/1.7.0/css/select.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/searchbuilder/1.6.0/css/searchBuilder.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/rowgroup/1.4.1/css/rowGroup.bootstrap5.min.css"
          rel="stylesheet">

    <!-- Chart.js for visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        body {{
            background-color: #f8f9fa;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        .container {{ max-width: 1400px; }}

        /* Modern card styling with subtle shadows and animations */
        .stat-card {{
            border: none;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.12);
        }}
        .stat-icon {{
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            font-size: 24px;
        }}

        /* Tab styling */
        .nav-tabs .nav-link {{
            color: #6c757d;
            border: none;
            padding: 12px 24px;
            font-weight: 500;
        }}
        .nav-tabs .nav-link.active {{
            color: #0d6efd;
            background-color: white;
            border-bottom: 3px solid #0d6efd;
        }}

        /* Module statistics modern styling */
        .module-stat-item {{
            border-left: 4px solid;
            transition: background-color 0.2s;
        }}
        .module-stat-item:hover {{
            background-color: rgba(0,0,0,0.02);
        }}

        /* Custom badge colors for categories */
        .badge.category-assert {{ background-color: #dc3545; }}
        .badge.category-value {{ background-color: #0dcaf0; }}
        .badge.category-type {{ background-color: #6f42c1; }}
        .badge.category-attr {{ background-color: #20c997; }}
        .badge.category-import {{ background-color: #ffc107; color: #000; }}
        .badge.category-index {{ background-color: #fd7e14; }}
        .badge.category-key {{ background-color: #6c757d; }}
        .badge.category-other {{ background-color: #e83e8c; }}

        /* Severity badges */
        .badge.severity-critical {{ background-color: #dc3545; }}
        .badge.severity-high {{ background-color: #fd7e14; }}
        .badge.severity-medium {{ background-color: #ffc107; color: #000; }}
        .badge.severity-low {{ background-color: #0d6efd; }}

        /* Status badges - using Bootstrap badge styling with proper contrast */
        .badge.status-passed {{
            background-color: #198754 !important;
            color: white !important;
            padding: 0.35em 0.65em;
            font-weight: 500;
        }}
        .badge.status-failed {{
            background-color: #dc3545 !important;
            color: white !important;
            padding: 0.35em 0.65em;
            font-weight: 500;
        }}
        .badge.status-skipped {{
            background-color: #ffc107 !important;
            color: #212529 !important;
            padding: 0.35em 0.65em;
            font-weight: 500;
        }}
        .badge.status-error {{
            background-color: #fd7e14 !important;
            color: white !important;
            padding: 0.35em 0.65em;
            font-weight: 500;
        }}
        .badge.status-xfailed {{
            background-color: #6c757d !important;
            color: white !important;
            padding: 0.35em 0.65em;
            font-weight: 500;
        }}
        .badge.status-xpassed {{
            background-color: #0dcaf0 !important;
            color: #212529 !important;
            padding: 0.35em 0.65em;
            font-weight: 500;
        }}

        /* Duration badges */
        .badge.duration-fast {{ background-color: #28a745; }}
        .badge.duration-medium {{ background-color: #ffc107; color: #000; }}
        .badge.duration-slow {{ background-color: #dc3545; }}

        /* Action button styling */
        .github-link {{
            background-color: #212529;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }}
        .github-link:hover {{ background-color: #495057; color: white; }}

        /* Traceback modal styling */
        .traceback-content {{
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 16px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.875rem;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }}

        /* Export dropdown styling */
        .dt-button-collection {{
            background: white;
            border: 1px solid rgba(0,0,0,0.15);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 0.5rem 0;
            min-width: 200px;
        }}

        /* Table container fixes */
        .dataTables_wrapper {{
            width: 100% !important;
        }}

        .dataTables_scrollBody {{
            overflow-x: auto !important;
        }}

        /* Ensure table stays within card boundaries */
        .card-body {{
            overflow-x: auto;
            padding: 1rem;
        }}


        /* Compact table cells for better fit */
        .table td, .table th {{
            padding: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* Error message column can wrap (9th column in failures table) */
        #failures-table td:nth-child(9) {{
            white-space: normal;
            word-wrap: break-word;
        }}

        /* Test name column can show ellipsis on overflow */
        #failures-table td:nth-child(7) {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* Row group styling */
        tr.dtrg-group {{
            background-color: #f8f9fa !important;
            font-weight: bold;
            cursor: pointer;
        }}

        tr.dtrg-group:hover {{
            background-color: #e9ecef !important;
        }}

        tr.dtrg-group td {{
            padding-top: 10px !important;
            padding-bottom: 10px !important;
            border-top: 2px solid #dee2e6;
        }}

        /* Group by selector styling */
        .group-by-selector {{
            margin-bottom: 1rem;
        }}

        .group-by-selector label {{
            font-weight: 500;
            margin-right: 0.5rem;
        }}

    </style>
</head>
<body>
    <div class="container py-4">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col">
                <h1 class="display-5 fw-bold text-primary">Datarax Test Results Dashboard</h1>
                <p class="text-muted">
                    Last updated: {timestamp} | Test Duration: {duration:.2f}s |
                    Exit Code: {exit_code}
                </p>
            </div>
        </div>

        <!-- Summary Cards with Modern Design -->
        <div class="row g-3 mb-4">
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <p class="text-muted small mb-1">Total Tests</p>
                                <h2 class="fw-bold mb-0">{total_count}</h2>
                            </div>
                            <div class="stat-icon bg-primary bg-opacity-10 text-primary">
                                📊
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <p class="text-muted small mb-1">Passed</p>
                                <h2 class="fw-bold mb-0 text-success">{passed_count}</h2>
                            </div>
                            <div class="stat-icon bg-success bg-opacity-10 text-success">
                                ✅
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <p class="text-muted small mb-1">Failed</p>
                                <h2 class="fw-bold mb-0 text-danger">{failed_count}</h2>
                            </div>
                            <div class="stat-icon bg-danger bg-opacity-10 text-danger">
                                ❌
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <p class="text-muted small mb-1">Skipped</p>
                                <h2 class="fw-bold mb-0 text-warning">{skipped_count}</h2>
                            </div>
                            <div class="stat-icon bg-warning bg-opacity-10 text-warning">
                                ⏭️
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <p class="text-muted small mb-1">Errors</p>
                                <h2 class="fw-bold mb-0 text-danger">{error_count}</h2>
                            </div>
                            <div class="stat-icon bg-danger bg-opacity-10 text-danger">
                                ⚠️
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div>
                            <p class="text-muted small mb-1">Success Rate</p>
                            <div class="d-flex align-items-center">
                                <div class="progress flex-grow-1" style="height: 8px;">
                                    <div class="progress-bar bg-success"
                                         style="width: {success_rate:.1f}%"></div>
                                </div>
                                <span class="ms-2 fw-bold text-success">{success_rate:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tabs Navigation -->
        <ul class="nav nav-tabs mb-4" role="tablist">
            <li class="nav-item">
                <a class="nav-link active" data-bs-toggle="tab" href="#failures-tab" role="tab">
                    <i class="fas fa-exclamation-triangle"></i> Failures Analysis
                    <span class="badge bg-danger ms-2">{total_failures}</span>
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#performance-tab" role="tab">
                    <i class="fas fa-clock"></i> Performance
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#modules-tab" role="tab">
                    <i class="fas fa-folder-tree"></i> Module Analysis
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#all-tests-tab" role="tab">
                    <i class="fas fa-list"></i> All Tests
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#coverage-tab" role="tab">
                    <i class="fas fa-chart-pie"></i> Coverage
                    {coverage_badge}
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#instructions-tab" role="tab">
                    <i class="fas fa-info-circle"></i> Instructions
                </a>
            </li>
        </ul>

        <!-- Tab Content -->
        <div class="tab-content">
            <!-- Failures Tab -->
            <div class="tab-pane fade show active" id="failures-tab" role="tabpanel">
                <div class="card">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">Failed Tests</h5>
                            <div class="group-by-selector">
                                <label for="groupBySelect">Group By:</label>
                                <select id="groupBySelect"
                                        class="form-select form-select-sm d-inline-block"
                                        style="width: auto;">
                                    <option value="none">None</option>
                                    <option value="7">Module</option>
                                    <option value="3">Error Type</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <table id="failures-table" class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Test ID</th>
                                    <th>Actions</th>
                                    <th>Trace</th>
                                    <th>Error Type</th>
                                    <th>Severity</th>
                                    <th>Duration</th>
                                    <th>Test Name</th>
                                    <th>Module</th>
                                    <th>Error Message</th>
                                    <th>Test Location</th>
                                </tr>
                            </thead>
                            <tbody>
                                {failures_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Performance Tab -->
            <div class="tab-pane fade" id="performance-tab" role="tabpanel">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Performance Analysis</h5>
                    </div>
                    <div class="card-body">
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <canvas id="durationChart"></canvas>
                            </div>
                            <div class="col-md-6">
                                <h6>Performance Stats</h6>
                                <dl class="row">
                                    <dt class="col-sm-6">Total Duration:</dt>
                                    <dd class="col-sm-6">{duration:.2f}s</dd>
                                    <dt class="col-sm-6">Average Test Time:</dt>
                                    <dd class="col-sm-6">{avg_duration:.3f}s</dd>
                                    <dt class="col-sm-6">Fast Tests (&lt;0.1s):</dt>
                                    <dd class="col-sm-6">{fast_tests}</dd>
                                    <dt class="col-sm-6">Medium Tests (0.1-1s):</dt>
                                    <dd class="col-sm-6">{medium_tests}</dd>
                                    <dt class="col-sm-6">Slow Tests (&gt;1s):</dt>
                                    <dd class="col-sm-6">{slow_tests}</dd>
                                </dl>
                            </div>
                        </div>
                        <table id="performance-table" class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Test Name</th>
                                    <th>Duration</th>
                                    <th>Speed</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {performance_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Modules Tab -->
            <div class="tab-pane fade" id="modules-tab" role="tabpanel">
                <div class="card">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">Module Statistics</h5>
                            <div>
                                <label for="moduleSortSelect" class="me-2">Sort by:</label>
                                <select id="moduleSortSelect"
                                        class="form-select form-select-sm d-inline-block"
                                        style="width: auto;">
                                    <option value="5-desc">Pass Rate (High to Low)</option>
                                    <option value="5-asc">Pass Rate (Low to High)</option>
                                    <option value="0-asc">Module Name (A-Z)</option>
                                    <option value="0-desc">Module Name (Z-A)</option>
                                    <option value="1-desc">Total Tests (High to Low)</option>
                                    <option value="3-desc">Failed Tests (High to Low)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="module-stats-container">
                            <table id="module-stats-table" class="table" style="display: none;">
                                <thead>
                                    <tr>
                                        <th>Module</th>
                                        <th>Total</th>
                                        <th>Passed</th>
                                        <th>Failed</th>
                                        <th>Skipped</th>
                                        <th>Pass Rate</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {module_stats_rows}
                                </tbody>
                            </table>
                            <div id="module-cards-container" class="list-group">
                                <!-- Cards will be rendered here by DataTables -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- All Tests Tab -->
            <div class="tab-pane fade" id="all-tests-tab" role="tabpanel">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">All Tests</h5>
                    </div>
                    <div class="card-body">
                        <table id="all-tests-table" class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Actions</th>
                                    <th>Test Name</th>
                                    <th>Status</th>
                                    <th>Duration</th>
                                    <th>Module</th>
                                </tr>
                            </thead>
                            <tbody>
                                {all_tests_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Coverage Tab -->
            <div class="tab-pane fade" id="coverage-tab" role="tabpanel">
                <div class="row mb-4">
                    <!-- Overall Coverage Stats -->
                    <div class="col-lg-3 col-md-6">
                        <div class="card stat-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <p class="text-muted small mb-1">Line Coverage</p>
                                        <h2 class="fw-bold mb-0">{line_coverage:.1f}%</h2>
                                        <small class="text-muted">
                                            {covered_lines}/{total_statements} lines
                                        </small>
                                    </div>
                                    <div class="stat-icon bg-primary bg-opacity-10 text-primary">
                                        📊
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="card stat-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <p class="text-muted small mb-1">Branch Coverage</p>
                                        <h2 class="fw-bold mb-0">{branch_coverage:.1f}%</h2>
                                        <small class="text-muted">
                                            {covered_branches}/{total_branches} branches
                                        </small>
                                    </div>
                                    <div class="stat-icon bg-info bg-opacity-10 text-info">
                                        🌿
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="card stat-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <p class="text-muted small mb-1">Missing Lines</p>
                                        <h2 class="fw-bold mb-0 text-danger">{missing_lines}</h2>
                                        <small class="text-muted">Lines not covered</small>
                                    </div>
                                    <div class="stat-icon bg-danger bg-opacity-10 text-danger">
                                        ⚠️
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-3 col-md-6">
                        <div class="card stat-card">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <p class="text-muted small mb-1">Files Analyzed</p>
                                        <h2 class="fw-bold mb-0">{files_count}</h2>
                                        <small class="text-muted">Source files</small>
                                    </div>
                                    <div class="stat-icon bg-success bg-opacity-10 text-success">
                                        📁
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Module Coverage Analysis</h5>
                    </div>
                    <div class="card-body">
                        <table id="coverage-table" class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Module</th>
                                    <th>Statements</th>
                                    <th>Covered</th>
                                    <th>Missing</th>
                                    <th>Coverage %</th>
                                    <th>Branches</th>
                                    <th>Covered Branches</th>
                                    <th>Missing Branches</th>
                                    <th>File Path</th>
                                </tr>
                            </thead>
                            <tbody>
                                {coverage_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Instructions Tab -->
            <div class="tab-pane fade" id="instructions-tab" role="tabpanel">
                <div class="card">
                    <div class="card-body">
                        <h5>How to Generate Test Reports</h5>
                        <p>This dashboard analyzes JSON test reports generated by
                           pytest-json-report.</p>

                        <h6 class="mt-4">1. Install Required Package</h6>
                        <pre><code>uv add --dev pytest-json-report</code></pre>

                        <h6 class="mt-4">2. Generate Test Report</h6>
                        <pre><code># Basic report
uv run pytest --json-report --json-report-file=temp/test-results.json

# With coverage only
uv run pytest --json-report --json-report-file=temp/test-results.json \\
    --cov=src/datarax --cov-report=json:temp/coverage.json

# With verbosity only
uv run pytest -vv --json-report --json-report-file=temp/test-results.json \\
    --json-report-indent=2 --json-report-verbosity=2

# Complete command with both coverage and verbosity (RECOMMENDED)
uv run pytest -vv \\
    --json-report --json-report-file=temp/datarax-test-results.json \\
    --json-report-indent=2 --json-report-verbosity=2 \\
    --cov=src/datarax \\
    --cov-report=json:temp/coverage.json \\
    --cov-report=term-missing

# For Datarax project specifically (exact command for all data)
uv run pytest -vv \\
    --json-report --json-report-file=temp/datarax-test-results.json \\
    --cov=src/datarax \\
    --cov-report=json:temp/coverage.json \\
    --cov-report=html:temp/coverage_html \\
    --cov-report=term-missing:skip-covered</code></pre>

                        <h6 class="mt-4">3. Generate Dashboard</h6>
                        <pre><code># Basic dashboard without coverage
python tests/scripts/test_results_analyzer.py \\
    --json-report temp/test-results.json \\
    --output temp/test_results_dashboard.html

# Full dashboard with coverage analysis (recommended)
python tests/scripts/test_results_analyzer.py \\
    --json-report temp/test-results.json \\
    --output temp/test_results_dashboard.html \\
    --coverage temp/coverage.json

# For Datarax project specifically (exact command used)
uv run python tests/scripts/test_results_analyzer.py \\
    --json-report temp/datarax-test-results.json \\
    --output temp/test_results_dashboard.html \\
    --coverage temp/coverage.json</code></pre>

                        <h6 class="mt-4">4. Dashboard Features</h6>
                        <div class="alert alert-info">
                            <h6>Filtering & Sorting Options:</h6>
                            <ul>
                                <li><strong>Failures Analysis Tab:</strong>
                                    <ul>
                                        <li>Group By: Module or Error Type with
                                            expandable/collapsible groups</li>
                                        <li>Advanced Filters: Click to filter by multiple
                                            criteria with range support</li>
                                        <li>Column Sorting: Click any column header to sort</li>
                                        <li>Search: Global text search across all columns</li>
                                    </ul>
                                </li>
                                <li><strong>Module Statistics Tab:</strong>
                                    <ul>
                                        <li>Sort By: Pass Rate, Module Name, Total Tests,
                                            Failed Tests</li>
                                        <li>Search: Filter modules by name using the search box</li>
                                    </ul>
                                </li>
                                <li><strong>Coverage Tab:</strong>
                                    <ul>
                                        <li>Advanced Filters: Filter by coverage
                                            percentage ranges</li>
                                        <li>Export: CSV, Excel, PDF formats available</li>
                                    </ul>
                                </li>
                                <li><strong>All Tabs Support:</strong>
                                    <ul>
                                        <li>Export buttons for CSV, Excel, PDF, Print</li>
                                        <li>Responsive pagination controls</li>
                                        <li>Real-time search and filtering</li>
                                    </ul>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Traceback Modal -->
    <div class="modal fade" id="tracebackModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Test Failure Traceback</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div id="traceback-content" class="traceback-content"></div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="copyTraceback()">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                    <button type="button" class="btn btn-primary"
                            data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- DataTables JS -->
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>

    <!-- DataTables Extensions - Order matters! -->
    <!-- Select MUST be loaded before SearchPanes -->
    <script src="https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js"></script>

    <!-- Buttons extension -->
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.bootstrap5.min.js"></script>

    <!-- Export button dependencies -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/pdfmake.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/vfs_fonts.js"></script>

    <!-- Export buttons -->
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.print.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>

    <!-- SearchPanes MUST be loaded after Select -->
    <script src="https://cdn.datatables.net/searchpanes/2.2.0/js/dataTables.searchPanes.min.js">
    </script>
    <script src="https://cdn.datatables.net/searchpanes/2.2.0/js/searchPanes.bootstrap5.min.js">
    </script>

    <!-- SearchBuilder for advanced filtering with range support -->
    <script src="https://cdn.datatables.net/searchbuilder/1.6.0/js/dataTables.searchBuilder.min.js">
    </script>
    <script src="https://cdn.datatables.net/searchbuilder/1.6.0/js/searchBuilder.bootstrap5.min.js">
    </script>

    <!-- RowGroup for grouping functionality -->
    <script src="https://cdn.datatables.net/rowgroup/1.4.1/js/dataTables.rowGroup.min.js"></script>

    <script>
        // Store tracebacks
        const tracebacks = {tracebacks_json};

        function showTraceback(testId) {{
            const content = tracebacks[testId] || 'No traceback available';
            document.getElementById('traceback-content').innerText = content;
            new bootstrap.Modal(document.getElementById('tracebackModal')).show();
        }}

        function copyTraceback() {{
            const text = document.getElementById('traceback-content').innerText;
            navigator.clipboard.writeText(text).then(() => {{
                alert('Traceback copied to clipboard!');
            }});
        }}

        // Initialize DataTables when DOM is ready
        $(document).ready(function() {{
            console.log('Initializing DataTables...');

            // Initialize Failures DataTable with SearchBuilder and RowGroup
            var failuresTable = $('#failures-table').DataTable({{
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                order: [[4, 'desc']],  // Sort by severity by default
                dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                     '<"row"<"col-sm-12"B>>' +
                     '<"row"<"col-sm-12"Q>>' +  // Q for SearchBuilder
                     '<"row"<"col-sm-12"tr>>' +
                     '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
                columnDefs: [
                    {{ type: 'num', targets: [5] }}  // Duration column is numeric
                ],
                buttons: [
                    {{
                        extend: 'searchBuilder',
                        text: 'Advanced Filters',
                        className: 'btn btn-primary'
                    }},
                    {{
                        extend: 'collection',
                        text: 'Export',
                        className: 'btn btn-success',
                        buttons: ['copy', 'csv', 'excel', 'pdf', 'print']
                    }}
                ],
                rowGroup: {{
                    enable: false,
                    dataSrc: 7,  // Default to Module column
                    startRender: function(rows, group) {{
                        // Initialize collapsed state for new groups (default to collapsed)
                        if (!(group in collapsedGroups)) {{
                            collapsedGroups[group] = true;
                        }}

                        var collapsed = collapsedGroups[group];

                        rows.nodes().each(function(r) {{
                            r.style.display = collapsed ? 'none' : '';
                        }});

                        var count = rows.count();
                        return $('<tr/>')
                            .append('<td colspan="10">' +
                                '<span class="group-toggle" style="cursor: pointer;">' +
                                    (collapsed ? '▶ ' : '▼ ') +
                                '</span>' +
                                '<strong>' + group + '</strong> ' +
                                '<span class="badge bg-secondary">' + count + ' failures</span>' +
                            '</td>')
                            .attr('data-name', group)
                            .toggleClass('collapsed', collapsed);
                    }}
                }}
            }});

            // Track collapsed groups
            var collapsedGroups = {{}};

            // Handle group by selector
            $('#groupBySelect').on('change', function() {{
                var value = $(this).val();
                collapsedGroups = {{}};  // Reset collapsed state before changing grouping
                if (value === 'none') {{
                    // When disabling grouping, make sure all rows are visible
                    $('#failures-table tbody tr').each(function() {{
                        this.style.display = '';
                    }});
                    failuresTable.rowGroup().disable().draw();
                }} else {{
                    // When grouping is enabled, all groups will start collapsed
                    failuresTable.rowGroup().enable().dataSrc(parseInt(value)).draw();
                }}
            }});

            // Handle group expand/collapse
            $('#failures-table tbody').on('click', 'tr.dtrg-group', function() {{
                var name = $(this).attr('data-name');
                collapsedGroups[name] = !collapsedGroups[name];
                failuresTable.draw(false);
            }});
            console.log('Failures table initialized successfully');

            // Initialize Performance DataTable with SearchBuilder
            $('#performance-table').DataTable({{
                pageLength: 25,
                order: [[1, 'desc']],  // Sort by duration descending
                dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                     '<"row"<"col-sm-12"B>>' +
                     '<"row"<"col-sm-12"Q>>' +  // Q for SearchBuilder
                     '<"row"<"col-sm-12"tr>>' +
                     '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
                columnDefs: [
                    {{ type: 'num', targets: [1] }},  // Duration column
                    {{ type: 'string', targets: [2] }}  // Speed category is string
                ],
                buttons: [
                    {{
                        extend: 'searchBuilder',
                        text: 'Advanced Filters',
                        className: 'btn btn-primary'
                    }},
                    {{
                        extend: 'collection',
                        text: 'Export',
                        className: 'btn btn-success',
                        buttons: ['copy', 'csv', 'excel', 'pdf', 'print']
                    }}
                ]
            }});
            console.log('Performance table initialized successfully');

            // Initialize All Tests DataTable with SearchBuilder
            $('#all-tests-table').DataTable({{
                pageLength: 50,
                dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                     '<"row"<"col-sm-12"B>>' +
                     '<"row"<"col-sm-12"Q>>' +  // Q for SearchBuilder
                     '<"row"<"col-sm-12"tr>>' +
                     '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
                columnDefs: [
                    {{ type: 'num', targets: [3] }}  // Duration column is numeric
                ],
                buttons: [
                    {{
                        extend: 'searchBuilder',
                        text: 'Advanced Filters',
                        className: 'btn btn-primary'
                    }},
                    {{
                        extend: 'collection',
                        text: 'Export',
                        className: 'btn btn-success',
                        buttons: ['copy', 'csv', 'excel', 'pdf', 'print']
                    }}
                ]
            }});
            console.log('All tests table initialized successfully');

            // Initialize Module Statistics DataTable with card rendering
            var moduleStatsTable = $('#module-stats-table').DataTable({{
                pageLength: -1,  // Show all modules
                dom: '<"row"<"col-sm-12 col-md-6"f><"col-sm-12 col-md-6"<"float-end"l>>>' +
                     '<"row"<"col-sm-12"<"module-cards-wrapper">>>' +
                     '<"row"<"col-sm-12"i>>',
                order: [[5, 'desc']],  // Sort by pass rate descending by default
                columnDefs: [
                    {{ type: 'num', targets: [1, 2, 3, 4, 5] }}
                ],
                language: {{
                    search: "Search modules:",
                    lengthMenu: "Show _MENU_ modules",
                    info: "Showing _TOTAL_ modules"
                }},
                drawCallback: function() {{
                    var api = this.api();
                    var container = $('#module-cards-container');
                    container.empty();

                    api.rows({{ page: 'current' }}).every(function() {{
                        var data = this.data();
                        var module = data[0];
                        var total = parseInt(data[1]);
                        var passed = parseInt(data[2]);
                        var failed = parseInt(data[3]);
                        var skipped = parseInt(data[4]);
                        var passed_pct = parseFloat(data[5]);

                        // Handle NaN case
                        if (isNaN(passed_pct)) {{
                            passed_pct = 0;
                        }}

                        // Determine color based on pass rate
                        var colorClass = 'danger';
                        if (passed_pct >= 90) colorClass = 'success';
                        else if (passed_pct >= 70) colorClass = 'info';
                        else if (passed_pct >= 50) colorClass = 'warning';

                        var card = `
                            <div class="list-group-item module-stat-item
                                 border-${{colorClass}} py-3">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <h6 class="mb-1 fw-semibold">${{module}}</h6>
                                        <div class="d-flex flex-wrap gap-2 mt-2">
                                            <span class="badge bg-primary">Total: ${{total}}</span>
                                            <span class="badge bg-success">
                                                Passed: ${{passed}} (${{passed_pct}}%)
                                            </span>
                                            <span class="badge bg-danger">Failed: ${{failed}}</span>
                                            <span class="badge bg-warning">
                                                Skipped: ${{skipped}}
                                            </span>
                                        </div>
                                    </div>
                                    <div class="text-end">
                                        <div class="progress" style="width: 100px; height: 6px;">
                                            <div class="progress-bar bg-success"
                                                 style="width: ${{passed_pct}}%"></div>
                                        </div>
                                        <small class="text-muted">${{passed_pct}}% pass rate</small>
                                    </div>
                                </div>
                            </div>
                        `;
                        container.append(card);
                    }});
                }}
            }});
            console.log('Module stats table initialized successfully');

            // Handle module sort selector
            $('#moduleSortSelect').on('change', function() {{
                var value = $(this).val().split('-');
                var column = parseInt(value[0]);
                var direction = value[1];
                moduleStatsTable.order([column, direction]).draw();
            }});

            // Initialize Coverage DataTable with SearchBuilder for range filtering
            if ($('#coverage-table').length) {{
                $('#coverage-table').DataTable({{
                    pageLength: 25,
                    lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                    order: [[4, 'asc']],  // Sort by coverage percentage ascending (worst first)
                    dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                         '<"row"<"col-sm-12"B>>' +
                         '<"row"<"col-sm-12"Q>>' +  // Q for SearchBuilder
                         '<"row"<"col-sm-12"tr>>' +
                         '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',
                    columnDefs: [
                        {{ type: 'num', targets: [1, 2, 3, 4, 5, 6] }}  // All numeric columns
                    ],
                    buttons: [
                        {{
                            extend: 'searchBuilder',
                            text: 'Advanced Filters',
                            className: 'btn btn-primary'
                        }},
                        {{
                            extend: 'collection',
                            text: 'Export',
                            className: 'btn btn-success',
                            buttons: ['copy', 'csv', 'excel', 'pdf', 'print']
                        }}
                    ]
                }});
                console.log('Coverage table initialized successfully');
            }}

            // Duration distribution chart
            const ctx = document.getElementById('durationChart');
            if (ctx) {{
                new Chart(ctx.getContext('2d'), {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Fast (<0.1s)', 'Medium (0.1-1s)', 'Slow (>1s)'],
                        datasets: [{{
                            data: [{fast_tests}, {medium_tests}, {slow_tests}],
                            backgroundColor: ['#28a745', '#ffc107', '#dc3545']
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{
                            legend: {{ position: 'bottom' }}
                        }}
                    }}
                }});
            }}
        }});
    </script>
</body>
</html>
"""

TABLE_ROW_TEMPLATE = """
            <tr>
                <td>{test_id}</td>
                <td>
                    <button class="github-link"
                            onclick="window.open('{github_url}', '_blank')">View</button>
                </td>
                <td>
                    <button class="btn btn-sm btn-secondary"
                            onclick="showTraceback('{test_id}')">Trace</button>
                </td>
                <td><span class="badge category-{category_class}">{error_type}</span></td>
                <td><span class="badge severity-{severity_class}">{severity}</span></td>
                <td><span class="badge duration-{duration_class}">{duration:.3f}s</span></td>
                <td title="{test_name}">{test_name_short}</td>
                <td>{module_short}</td>
                <td title="{error_message}">{error_message_short}</td>
                <td title="{test_location}">{test_location_short}</td>
            </tr>
"""

MODULE_CARD_TEMPLATE = """
<div class="list-group-item module-stat-item border-{color_class} py-3">
    <div class="d-flex justify-content-between align-items-center">
        <div>
            <h6 class="mb-1 fw-semibold">{module_name}</h6>
            <div class="d-flex flex-wrap gap-2 mt-2">
                <span class="badge bg-primary">Total: {total}</span>
                <span class="badge bg-success">Passed: {passed} ({passed_pct}%)</span>
                <span class="badge bg-danger">Failed: {failed}</span>
                <span class="badge bg-warning">Skipped: {skipped}</span>
            </div>
        </div>
        <div class="text-end">
            <div class="progress" style="width: 100px; height: 6px;">
                <div class="progress-bar bg-success" style="width: {passed_pct}%"></div>
            </div>
            <small class="text-muted">{passed_pct}% pass rate</small>
        </div>
    </div>
</div>
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate detailed test results analysis dashboard"
    )
    parser.add_argument(
        "--json-report",
        type=str,
        default="temp/datarax-test-results.json",
        help="Path to pytest JSON report file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="temp/test_results_dashboard.html",
        help="Output path for the generated dashboard",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--coverage",
        type=str,
        default="temp/coverage.json",
        help="Path to coverage JSON file (optional)",
    )
    return parser.parse_args()


def load_json_report(report_path: str) -> dict[str, Any] | None:
    """Load and parse pytest JSON report."""
    if not os.path.exists(report_path):
        logger.error(f"Report file not found: {report_path}")
        logger.info(
            f"Generate report with: uv run pytest --json-report --json-report-file={report_path}"
        )
        return None

    with open(report_path, "r") as f:
        return json.load(f)


def load_coverage_data(coverage_path: str) -> dict[str, Any] | None:
    """Load and parse coverage JSON report."""
    if not coverage_path or not os.path.exists(coverage_path):
        logger.info(f"Coverage file not found: {coverage_path}")
        logger.info(
            "Generate coverage with: uv run pytest --cov=src/datarax "
            "--cov-report=json:temp/coverage.json"
        )
        return None

    try:
        with open(coverage_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load coverage data: {e}")
        return None


def categorize_error(error_message: str) -> tuple[str, str, str]:
    """Categorize error message and assign severity."""
    error_message.lower()

    # Extract error type
    if "AssertionError" in error_message:
        category = "assert"
        error_type = "AssertionError"
        severity = "medium"
    elif "ValueError" in error_message:
        category = "value"
        error_type = "ValueError"
        severity = "high"
    elif "TypeError" in error_message:
        category = "type"
        error_type = "TypeError"
        severity = "high"
    elif "AttributeError" in error_message:
        category = "attr"
        error_type = "AttributeError"
        severity = "high"
    elif "ImportError" in error_message or "ModuleNotFoundError" in error_message:
        category = "import"
        error_type = "ImportError"
        severity = "critical"
    elif "IndexError" in error_message:
        category = "index"
        error_type = "IndexError"
        severity = "medium"
    elif "KeyError" in error_message:
        category = "key"
        error_type = "KeyError"
        severity = "medium"
    else:
        category = "other"
        error_type = error_message.split(":")[0] if ":" in error_message else "Error"
        severity = "low"

    return category, error_type, severity


def get_duration_class(duration: float) -> str:
    """Get duration class for styling."""
    if duration < 0.1:
        return "fast"
    elif duration < 1.0:
        return "medium"
    else:
        return "slow"


def get_github_url(test_location: str, line_number: int | None = None) -> str:
    """Generate a GitHub URL for a test file with optional line number."""
    try:
        repo_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]

        if "github.com" in repo_url:
            if repo_url.startswith("git@github.com:"):
                owner_repo = repo_url.split("git@github.com:")[1]
            else:
                owner_repo = repo_url.split("github.com/")[1]

            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                universal_newlines=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            # Parse test location
            if "::" in test_location:
                file_path = test_location.split("::")[0]
            else:
                file_path = test_location

            # Build URL with optional line number
            base_url = f"https://github.com/{owner_repo}/blob/{branch}/{file_path}"
            if line_number:
                return f"{base_url}#L{line_number}"
            return base_url
    except Exception:
        pass

    return "#"


def process_test_results(report_data: dict) -> dict[str, Any]:
    """Process test results and extract relevant information."""
    tests = report_data.get("tests", [])
    summary = report_data.get("summary", {})

    failed_tests = []
    performance_data = []
    all_tests = []
    tracebacks = {}

    for test in tests:
        nodeid = test.get("nodeid", "")
        outcome = test.get("outcome", "")
        line_number = test.get("lineno")  # Get the line number where test is defined

        # Extract module
        if "::" in nodeid:
            module = nodeid.split("::")[0].replace("/", ".").replace(".py", "")
            test_name = nodeid.split("::")[-1]
        else:
            module = "unknown"
            test_name = nodeid

        # Generate unique ID
        test_id = hashlib.md5(nodeid.encode()).hexdigest()[:8].upper()

        # Get duration
        duration = test.get("call", {}).get("duration", 0) if "call" in test else 0

        # Process all tests
        all_tests.append(
            {
                "test_id": test_id,
                "nodeid": nodeid,
                "test_name": test_name,
                "module": module,
                "outcome": outcome,
                "duration": duration,
                "github_url": get_github_url(nodeid, line_number),
            }
        )

        # Process performance data
        if duration > 0:
            performance_data.append(
                {
                    "nodeid": nodeid,
                    "test_name": test_name,
                    "duration": duration,
                    "outcome": outcome,
                    "duration_class": get_duration_class(duration),
                }
            )

        # Process failed tests
        if outcome in ["failed", "error"]:
            error_message = ""
            error_type = "Unknown"
            category = "other"
            severity = "low"

            if "call" in test and "crash" in test["call"]:
                error_message = test["call"]["crash"].get("message", "")
                category, error_type, severity = categorize_error(error_message)

                # Store full traceback
                if "longrepr" in test["call"]:
                    tracebacks[test_id] = test["call"]["longrepr"]
                elif "traceback" in test["call"]["crash"]:
                    tracebacks[test_id] = "\n".join(
                        [
                            f"{tb.get('path', '')}:{tb.get('lineno', '')}: {tb.get('message', '')}"
                            for tb in test["call"]["crash"]["traceback"]
                        ]
                    )

            failed_tests.append(
                {
                    "test_id": test_id,
                    "nodeid": nodeid,
                    "test_name": test_name,
                    "module": module,
                    "error_message": error_message,
                    "error_type": error_type,
                    "category": category,
                    "severity": severity,
                    "duration": duration,
                    "github_url": get_github_url(nodeid, line_number),
                }
            )

    # Sort performance data by duration
    performance_data.sort(key=lambda x: x["duration"], reverse=True)

    return {
        "failed_tests": failed_tests,
        "performance_data": performance_data,
        "all_tests": all_tests,
        "tracebacks": tracebacks,
        "summary": summary,
    }


def generate_module_stats(tests: list[dict]) -> str:
    """Generate module statistics table rows for DataTable."""
    module_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0}
    )

    for test in tests:
        module = test["module"]
        outcome = test["outcome"]
        module_stats[module]["total"] += 1
        if outcome == "passed":
            module_stats[module]["passed"] += 1
        elif outcome == "failed":
            module_stats[module]["failed"] += 1
        elif outcome == "skipped":
            module_stats[module]["skipped"] += 1
        elif outcome == "error":
            module_stats[module]["error"] += 1

    rows = []
    for module, stats in sorted(module_stats.items()):
        passed_pct = round(stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        failed_total = stats["failed"] + stats["error"]

        rows.append(f"""
            <tr>
                <td>{module or "root"}</td>
                <td>{stats["total"]}</td>
                <td>{stats["passed"]}</td>
                <td>{failed_total}</td>
                <td>{stats["skipped"]}</td>
                <td>{passed_pct}</td>
            </tr>
        """)

    return "".join(rows)


def process_coverage_data(coverage_data: dict | None) -> dict[str, Any]:
    """Process coverage data for display."""
    if not coverage_data:
        return {
            "has_coverage": False,
            "totals": {},
            "files": [],
            "coverage_html": "<p>No coverage data available</p>",
            "coverage_rows": "",
        }

    totals = coverage_data.get("totals", {})
    files_data = coverage_data.get("files", {})

    # Process file coverage data
    coverage_files = []
    for file_path, file_info in files_data.items():
        # Convert path to module name
        if file_path.startswith("src/"):
            module = file_path[4:].replace("/", ".").replace(".py", "")
        else:
            module = file_path.replace("/", ".").replace(".py", "")

        summary = file_info.get("summary", {})
        coverage_files.append(
            {
                "file_path": file_path,
                "module": module,
                "statements": summary.get("num_statements", 0),
                "covered": summary.get("covered_lines", 0),
                "missing": summary.get("missing_lines", 0),
                "coverage_pct": summary.get("percent_covered", 0),
                "branches": summary.get("num_branches", 0),
                "covered_branches": summary.get("covered_branches", 0),
                "missing_branches": summary.get("missing_branches", 0),
            }
        )

    # Sort by coverage percentage (ascending so worst are at top)
    coverage_files.sort(key=lambda x: x["coverage_pct"])

    # Generate coverage table rows
    coverage_rows = []
    for file_data in coverage_files:
        # Determine coverage color class
        coverage_pct = file_data["coverage_pct"]
        if coverage_pct >= 90:
            coverage_class = "success"
        elif coverage_pct >= 70:
            coverage_class = "warning"
        elif coverage_pct >= 50:
            coverage_class = "danger"
        else:
            coverage_class = "dark"

        coverage_rows.append(f"""
        <tr>
            <td>{file_data["module"]}</td>
            <td>{file_data["statements"]}</td>
            <td>{file_data["covered"]}</td>
            <td>{file_data["missing"]}</td>
            <td>
                <div class="progress" style="height: 20px; min-width: 100px;">
                    <div class="progress-bar bg-{coverage_class}"
                         style="width: {coverage_pct:.1f}%">
                        {coverage_pct:.1f}%
                    </div>
                </div>
            </td>
            <td>{file_data["branches"]}</td>
            <td>{file_data["covered_branches"]}</td>
            <td>{file_data["missing_branches"]}</td>
            <td>{file_data["file_path"]}</td>
        </tr>
        """)

    return {
        "has_coverage": True,
        "totals": totals,
        "files": coverage_files,
        "coverage_rows": "".join(coverage_rows),
    }


def generate_dashboard(
    report_data: dict, output_path: str, coverage_data: dict | None = None
) -> None:
    """Generate the HTML dashboard."""
    if not report_data:
        return

    # Process test results
    processed = process_test_results(report_data)

    # Get summary stats
    summary = processed["summary"]
    total_count = summary.get("total", 0)
    passed_count = summary.get("passed", 0)
    failed_count = summary.get("failed", 0)
    skipped_count = summary.get("skipped", 0)
    error_count = summary.get("error", 0)

    success_rate = (passed_count / total_count * 100) if total_count > 0 else 0
    total_failures = failed_count + error_count

    # Calculate performance stats
    performance_data = processed["performance_data"]
    total_duration = report_data.get("duration", 0)
    avg_duration = total_duration / len(performance_data) if performance_data else 0

    fast_tests = sum(1 for t in performance_data if t["duration"] < 0.1)
    medium_tests = sum(1 for t in performance_data if 0.1 <= t["duration"] < 1.0)
    slow_tests = sum(1 for t in performance_data if t["duration"] >= 1.0)

    # Generate failure rows
    failures_rows = []
    for test in processed["failed_tests"]:
        # Shorten long strings for display
        error_msg_short = (
            test["error_message"][:50] + "..."
            if len(test["error_message"]) > 50
            else test["error_message"]
        )
        test_name_short = (
            test["test_name"][:30] + "..." if len(test["test_name"]) > 30 else test["test_name"]
        )
        module_short = test["module"].split(".")[-1] if "." in test["module"] else test["module"]
        test_location = test["nodeid"].split("::")[0] if "::" in test["nodeid"] else test["nodeid"]
        test_location_short = (
            test_location.split("/")[-1] if "/" in test_location else test_location
        )

        failures_rows.append(
            TABLE_ROW_TEMPLATE.format(
                test_id=test["test_id"],
                test_name=test["test_name"],
                test_name_short=test_name_short,
                error_message=test["error_message"].replace('"', "&quot;"),
                error_message_short=error_msg_short,
                error_type=test["error_type"],
                category_class=test["category"],
                module=test["module"],
                module_short=module_short,
                duration=test["duration"],
                duration_class=get_duration_class(test["duration"]),
                severity=test["severity"].title(),
                severity_class=test["severity"],
                test_location=test_location,
                test_location_short=test_location_short,
                github_url=test["github_url"],
            )
        )

    # Generate performance rows
    performance_rows = []
    for test in performance_data[:100]:  # Show top 100
        performance_rows.append(f"""
        <tr>
            <td>{test["test_name"]}</td>
            <td><span class="badge duration-{test["duration_class"]}">
                {test["duration"]:.3f}s</span></td>
            <td><span class="badge duration-{test["duration_class"]}">
                {test["duration_class"].title()}</span></td>
            <td><span class="badge status-{test["outcome"]}">{test["outcome"]}</span></td>
        </tr>
        """)

    # Generate all tests rows
    all_tests_rows = []
    for test in processed["all_tests"][:500]:  # Limit to 500 for performance
        all_tests_rows.append(f"""
        <tr>
            <td>
                <button class="github-link"
                        onclick="window.open('{test["github_url"]}', '_blank')">View</button>
            </td>
            <td>{test["test_name"]}</td>
            <td><span class="badge status-{test["outcome"]}">{test["outcome"]}</span></td>
            <td>{test["duration"]:.3f}s</td>
            <td>{test["module"]}</td>
        </tr>
        """)

    # Generate module statistics table rows
    module_stats_rows = generate_module_stats(processed["all_tests"])

    # Process coverage data
    coverage_info = process_coverage_data(coverage_data)

    # Extract coverage statistics
    if coverage_info["has_coverage"]:
        totals = coverage_info["totals"]
        line_coverage = totals.get("percent_covered", 0)
        covered_lines = totals.get("covered_lines", 0)
        total_statements = totals.get("num_statements", 0)
        missing_lines = totals.get("missing_lines", 0)
        total_branches = totals.get("num_branches", 0)
        covered_branches = totals.get("covered_branches", 0)
        branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0
        files_count = len(coverage_info["files"])
        badge_class = (
            "success" if line_coverage >= 80 else "warning" if line_coverage >= 60 else "danger"
        )
        coverage_badge = f'<span class="badge bg-{badge_class} ms-2">{line_coverage:.0f}%</span>'
    else:
        line_coverage = 0
        covered_lines = 0
        total_statements = 0
        missing_lines = 0
        total_branches = 0
        covered_branches = 0
        branch_coverage = 0
        files_count = 0
        coverage_badge = '<span class="badge bg-secondary ms-2">N/A</span>'

    # Generate final HTML
    html = DASHBOARD_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration=total_duration,
        exit_code=report_data.get("exitcode", "N/A"),
        total_count=total_count,
        passed_count=passed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        error_count=error_count,
        success_rate=success_rate,
        total_failures=total_failures,
        avg_duration=avg_duration,
        fast_tests=fast_tests,
        medium_tests=medium_tests,
        slow_tests=slow_tests,
        failures_rows="".join(failures_rows),
        performance_rows="".join(performance_rows),
        all_tests_rows="".join(all_tests_rows),
        module_stats_rows=module_stats_rows,
        tracebacks_json=json.dumps(processed["tracebacks"]),
        # Coverage data
        coverage_rows=coverage_info["coverage_rows"],
        coverage_badge=coverage_badge,
        line_coverage=line_coverage,
        covered_lines=covered_lines,
        total_statements=total_statements,
        missing_lines=missing_lines,
        branch_coverage=branch_coverage,
        covered_branches=covered_branches,
        total_branches=total_branches,
        files_count=files_count,
    )

    # Write to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard generated at: {output_path}")
    logger.info(
        f"Summary: {passed_count} passed, {failed_count} failed, "
        f"{skipped_count} skipped, {error_count} errors"
    )


def main():
    """Main execution function."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load test report
    report_data = load_json_report(args.json_report)
    if not report_data:
        return 1

    # Load coverage data
    coverage_data = load_coverage_data(args.coverage)

    # Generate dashboard
    generate_dashboard(report_data, args.output, coverage_data)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
