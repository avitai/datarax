#!/usr/bin/env python
"""Skipped Tests Tracker - Comprehensive Test Skip Management System

PURPOSE:
--------
This script provides a complete solution for tracking, managing, and visualizing
skipped tests in your Python test suite. It automatically scans for pytest skip
decorators, categorizes them intelligently, tracks fixing progress over time,
and generates both human-readable dashboards and machine-readable reports.

FEATURES:
---------
1. Automatic Test Discovery:
   - Scans for pytest.skip(), @pytest.mark.skip, @pytest.mark.skipif, @pytest.mark.xfail
   - Extracts test names using AST parsing
   - Captures skip reasons and conditions
   - Generates unique IDs for tracking

2. Intelligent Categorization:
   - HW: Hardware-related (GPU/TPU requirements)
   - DEP: External dependency issues
   - ISS: Known implementation issues
   - FLAKY: Intermittent/flaky tests
   - PERF: Performance-related skips
   - ENV: Environment-specific issues
   - SEL: Selection/filtering issues
   - SEG: Segmentation/memory issues

3. Priority Assignment:
   - Priority 1: Critical issues (bugs, segfaults)
   - Priority 2: Dependencies blocking functionality
   - Priority 3: Standard issues
   - Priority 4: Environment/platform specific
   - Priority 5: Low priority/cosmetic

4. Progress Tracking:
   - Status management (To Fix, In Progress, Fixed, Won't Fix)
   - Historical tracking with trend analysis
   - Module-level statistics
   - Progress visualization

5. Multiple Output Formats:
   - HTML Dashboard: Interactive web interface with filtering
   - Markdown Inventory: Version-controlled documentation
   - JSON Export: CI/CD integration and programmatic access
   - History Tracking: Change detection over time

USAGE EXAMPLES:
---------------
1. Initial Setup - First Time Scan:
   ```bash
   # Scan codebase and create initial inventory
   python tests/scripts/skipped_tests_tracker.py --scan

   # Generate dashboard from inventory
   python tests/scripts/skipped_tests_tracker.py
   ```

2. Regular Workflow - Update and Track:
   ```bash
   # Re-scan and update inventory with new tests
   python tests/scripts/skipped_tests_tracker.py --scan

   # Update test status as you work
   python tests/scripts/skipped_tests_tracker.py --update-status TEST_ID "In Progress"
   python tests/scripts/skipped_tests_tracker.py --update-status TEST_ID "Fixed"
   ```

3. CI/CD Integration:
   ```bash
   # Generate JSON report for automated processing
   python tests/scripts/skipped_tests_tracker.py --scan --json ci/skipped_tests.json

   # Parse JSON in CI pipeline
   jq '.summary.by_status.to_fix' ci/skipped_tests.json
   ```

4. Custom Paths:
   ```bash
   # Use custom inventory location
   python tests/scripts/skipped_tests_tracker.py --inventory my_tests.md --scan

   # Generate dashboard in specific location
   python tests/scripts/skipped_tests_tracker.py --output reports/tests.html

   # Track history in custom location
   python tests/scripts/skipped_tests_tracker.py --history metrics/test_history.json
   ```

5. Verbose Mode for Debugging:
   ```bash
   python tests/scripts/skipped_tests_tracker.py --scan --verbose
   ```

COMMAND-LINE OPTIONS:
----------------------
--scan                  Scan codebase for skipped tests and update inventory
--inventory PATH        Path to inventory Markdown file (default: docs/skipped_tests_inventory.md)
--output PATH          Path for HTML dashboard output (default: temp/skipped_tests_dashboard.html)
--json PATH            Also generate JSON output for CI/CD integration
--history PATH         Path to history tracking file (default: temp/skipped_tests_history.json)
--update-status ID STATUS  Update the status of a specific test
                          STATUS must be one of: Fixed, In Progress, To Fix, Won't Fix
--verbose, -v          Enable verbose output for debugging

WORKFLOW SCENARIOS:
-------------------
Scenario 1: Project Onboarding
  1. Run initial scan: --scan
  2. Review dashboard to understand test debt
  3. Prioritize based on categories and priorities
  4. Create action plan for addressing skips

Scenario 2: Sprint Planning
  1. Generate fresh dashboard
  2. Filter by priority and category
  3. Assign "In Progress" status to sprint items
  4. Track completion throughout sprint

Scenario 3: CI/CD Quality Gate
  1. Run scan in CI pipeline
  2. Parse JSON output for metrics
  3. Fail build if skip count increases
  4. Generate reports for stakeholders

Scenario 4: Technical Debt Tracking
  1. Regular scans (weekly/monthly)
  2. Review history trends
  3. Report on debt reduction progress
  4. Identify problem areas (modules with many skips)

FILE FORMATS:
-------------
1. Inventory (Markdown):
   - Human-readable documentation
   - Version control friendly
   - Contains all test metadata
   - Auto-generated, manual edits preserved

2. Dashboard (HTML):
   - Interactive filtering and search
   - Visual progress indicators
   - Module statistics
   - Direct GitHub links

3. JSON Export:
   - Complete test data
   - Summary statistics
   - Aggregations by category/priority/module
   - Machine-readable for automation

4. History (JSON):
   - Timestamped snapshots
   - Trend calculations
   - Change detection
   - Progress over time

INTEGRATION:
------------
Pre-commit Hook:
  ```yaml
  - repo: local
    hooks:
      - id: check-skipped-tests
        name: Check Skipped Tests
        entry: python tests/scripts/skipped_tests_tracker.py --scan --json .metrics/skipped.json
        language: system
        always_run: true
  ```

GitHub Actions:
  ```yaml
  - name: Track Skipped Tests
    run: |
      python tests/scripts/skipped_tests_tracker.py --scan --json skipped.json
      echo "SKIPPED_COUNT=$(jq '.summary.total' skipped.json)" >> $GITHUB_ENV

  - name: Comment PR
    uses: actions/github-script@v6
    with:
      script: |
        github.rest.issues.createComment({
          issue_number: context.issue.number,
          body: `⚠️ This PR has ${process.env.SKIPPED_COUNT} skipped tests`
        })
  ```

BEST PRACTICES:
---------------
1. Regular Scanning:
   - Run --scan before each release
   - Include in CI/CD pipeline
   - Track trends weekly/monthly

2. Status Management:
   - Update status as work progresses
   - Use "Won't Fix" for legitimate skips
   - Document reasons in commit messages

3. Team Collaboration:
   - Share dashboard link in documentation
   - Review in sprint retrospectives
   - Set skip reduction targets

4. Version Control:
   - Commit inventory file (docs/skipped_tests_inventory.md)
   - Exclude dashboard HTML from version control
   - Track history file for long-term metrics

CUSTOMIZATION:
--------------
To add new categories or priorities, modify:
- categorize_skip_reason() function for category logic
- Category patterns dictionary for keywords
- HTML/CSS templates for visual styling

REQUIREMENTS:
-------------
- Python 3.8+
- pytest test framework
- Git (for GitHub URL generation)
- Modern web browser (for dashboard viewing)

AUTHOR: Datarax Development Team
LICENSE: Same as parent project
VERSION: 1.0.0
"""

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Datarax Skipped Tests Dashboard</title>

    <!-- jQuery (required for DataTables) -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>

    <!-- Bootstrap 5 CSS & JS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
          rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <!-- DataTables with Bootstrap 5 styling -->
    <link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/searchpanes/2.2.0/css/searchPanes.bootstrap5.min.css"
          rel="stylesheet">
    <link href="https://cdn.datatables.net/select/1.7.0/css/select.bootstrap5.min.css"
          rel="stylesheet">

    <style>
        /* Minimal custom styling - let Bootstrap and DataTables handle most styling */
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

        /* Module statistics modern styling */
        .module-stat-item {{
            border-left: 4px solid;
            transition: background-color 0.2s;
        }}
        .module-stat-item:hover {{
            background-color: rgba(0,0,0,0.02);
        }}

        /* Custom badge colors for categories and priorities */
        .badge.category-hw {{ background-color: #d63384; }}
        .badge.category-dep {{ background-color: #0dcaf0; }}
        .badge.category-iss {{ background-color: #dc3545; }}
        .badge.category-flaky {{ background-color: #20c997; }}
        .badge.category-perf {{ background-color: #6f42c1; }}
        .badge.category-env {{ background-color: #6c757d; }}
        .badge.category-sel {{ background-color: #fd7e14; }}
        .badge.category-seg {{ background-color: #e83e8c; }}

        .badge.priority-1 {{ background-color: #dc3545; }}
        .badge.priority-2 {{ background-color: #fd7e14; }}
        .badge.priority-3 {{ background-color: #ffc107; color: #000; }}
        .badge.priority-4 {{ background-color: #0d6efd; }}
        .badge.priority-5 {{ background-color: #6f42c1; }}

        .status.status-fixed {{ background-color: #d1e7dd; color: #0f5132; }}
        .status.status-in-progress {{ background-color: #cff4fc; color: #055160; }}
        .status.status-to-fix {{ background-color: #fff3cd; color: #664d03; }}
        .status.status-wont-fix {{ background-color: #f8d7da; color: #842029; }}

        /* Action button styling */
        .github-link {{
            background-color: #212529;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
        }}
        .github-link:hover {{ background-color: #495057; }}

        /* Export dropdown styling */
        .dt-button-collection {{
            background: white;
            border: 1px solid rgba(0,0,0,0.15);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 0.5rem 0;
            min-width: 200px;
        }}
        .dt-button-collection .dropdown-item {{
            padding: 0.5rem 1rem;
            transition: background-color 0.2s;
        }}
        .dt-button-collection .dropdown-item:hover {{
            background-color: #f8f9fa;
        }}
        .dt-button-collection .dropdown-item i {{
            width: 20px;
            margin-right: 8px;
            color: #6c757d;
        }}
        .dt-button-collection .dropdown-divider {{
            margin: 0.5rem 0;
        }}
    </style>
</head>
<body>
    <div class="container py-4">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col">
                <h1 class="display-5 fw-bold text-primary">Datarax Skipped Tests Dashboard</h1>
                <p class="text-muted">Last updated: {timestamp}</p>
            </div>
        </div>

        <!-- Summary Cards with Modern Design -->
        <div class="row g-3 mb-4">
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <p class="text-muted small mb-1">Total Skipped</p>
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
                                <p class="text-muted small mb-1">Fixed</p>
                                <h2 class="fw-bold mb-0 text-success">{fixed_count}</h2>
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
                                <p class="text-muted small mb-1">In Progress</p>
                                <h2 class="fw-bold mb-0 text-info">{in_progress_count}</h2>
                            </div>
                            <div class="stat-icon bg-info bg-opacity-10 text-info">
                                ⚡
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
                                <p class="text-muted small mb-1">To Fix</p>
                                <h2 class="fw-bold mb-0 text-warning">{to_fix_count}</h2>
                            </div>
                            <div class="stat-icon bg-warning bg-opacity-10 text-warning">
                                🔧
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
                                <p class="text-muted small mb-1">Won't Fix</p>
                                <h2 class="fw-bold mb-0 text-secondary">{wont_fix_count}</h2>
                            </div>
                            <div class="stat-icon bg-secondary bg-opacity-10 text-secondary">
                                🚫
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-2 col-md-4 col-sm-6">
                <div class="card stat-card">
                    <div class="card-body">
                        <div>
                            <p class="text-muted small mb-1">Overall Progress</p>
                            <div class="d-flex align-items-center">
                                <div class="progress flex-grow-1" style="height: 8px;">
                                    <div class="progress-bar bg-success"
                                         style="width: {progress}%"></div>
                                </div>
                                <span class="ms-2 fw-bold text-success">{progress:.0f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Module Stats Section -->
        <div class="row mb-4">
            <div class="col">
                {module_stats}
            </div>
        </div>


        <!-- Tests Table -->
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Skipped Tests</h5>
            </div>
            <div class="card-body">
                <table id="skipped-tests-table" class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Actions</th>
                        <th>ID</th>
                        <th>Test Name</th>
                        <th>Skip Reason</th>
                        <th>Category</th>
                        <th>Dependencies</th>
                        <th>Hardware</th>
                        <th>Status</th>
                        <th>Priority</th>
                        <th>Test Location</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Bootstrap 5 JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

    <!-- DataTables JS -->
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>

    <!-- DataTables Extensions (only essential ones) -->
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.bootstrap5.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/pdfmake.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.2.7/vfs_fonts.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.print.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.colVis.min.js"></script>
    <script src="https://cdn.datatables.net/select/1.7.0/js/dataTables.select.min.js"></script>
    <script src="https://cdn.datatables.net/searchpanes/2.2.0/js/dataTables.searchPanes.min.js"></script>
    <script src="https://cdn.datatables.net/searchpanes/2.2.0/js/searchPanes.bootstrap5.min.js"></script>

    <script>
        $(document).ready(function() {{
            // Initialize DataTables with modern SearchPanes
            const table = $('#skipped-tests-table').DataTable({{
                // Basic configuration
                pageLength: 25,
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],

                // Column definitions - must match HTML table order exactly
                columns: [
                    {{
                        title: "Actions",
                        orderable: false,
                        searchable: false,
                        className: "text-center"
                    }},
                    {{ title: "ID", className: "font-monospace" }},
                    {{ title: "Test Name" }},
                    {{ title: "Skip Reason" }},
                    {{
                        title: "Category",
                        className: "text-center"
                    }},
                    {{ title: "Dependencies" }},
                    {{ title: "Hardware" }},
                    {{
                        title: "Status",
                        className: "text-center"
                    }},
                    {{
                        title: "Priority",
                        className: "text-center",
                        type: "num"
                    }},
                    {{ title: "Test Location", className: "font-monospace small" }}
                ],

                // Enable SearchPanes and other features
                searching: true,
                ordering: true,
                info: true,
                autoWidth: false,
                responsive: true,

                // DOM layout with Bootstrap classes
                dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>' +
                     '<"row"<"col-sm-12 col-md-12"B>>' +
                     '<"row"<"col-sm-12"tr>>' +
                     '<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>',

                // SearchPanes configuration
                searchPanes: {{
                    cascadePanes: true,
                    viewTotal: true,
                    threshold: 0.1,  // Lower threshold to show more panes
                    layout: 'columns-4',
                    columns: [4, 6, 7, 8], // Category, Hardware, Status, Priority
                    initCollapsed: false,  // Show panes expanded when opened
                    dtOpts: {{
                        select: {{
                            style: 'multi'  // Allow multi-select in panes
                        }}
                    }}
                }},

                buttons: [
                    {{
                        extend: 'searchPanes',
                        text: '🔍 Advanced Filters',
                        titleAttr: 'Advanced Filters',
                        className: 'btn btn-primary'
                    }},
                    {{
                        extend: 'collection',
                        text: '📥 Export',
                        className: 'btn btn-success',
                        buttons: [
                            {{
                                extend: 'copy',
                                text: '<i class="far fa-copy"></i> Copy to Clipboard',
                                className: 'dropdown-item',
                                exportOptions: {{
                                    columns: ':visible'
                                }}
                            }},
                            {{
                                extend: 'csv',
                                text: '<i class="fas fa-file-csv"></i> Download CSV',
                                className: 'dropdown-item',
                                filename: 'skipped_tests_' + new Date().toISOString().split('T')[0],
                                exportOptions: {{
                                    columns: ':visible'
                                }}
                            }},
                            {{
                                extend: 'excel',
                                text: '<i class="fas fa-file-excel"></i> Download Excel',
                                className: 'dropdown-item',
                                filename: 'skipped_tests_' + new Date().toISOString().split('T')[0],
                                exportOptions: {{
                                    columns: ':visible'
                                }}
                            }},
                            {{
                                extend: 'pdf',
                                text: '<i class="fas fa-file-pdf"></i> Download PDF',
                                className: 'dropdown-item',
                                filename: 'skipped_tests_' + new Date().toISOString().split('T')[0],
                                orientation: 'landscape',
                                pageSize: 'A4',
                                exportOptions: {{
                                    columns: ':visible'
                                }}
                            }},
                            '<hr class="dropdown-divider">',
                            {{
                                extend: 'print',
                                text: '<i class="fas fa-print"></i> Print Preview',
                                className: 'dropdown-item',
                                exportOptions: {{
                                    columns: ':visible'
                                }}
                            }}
                        ],
                        autoClose: true,
                        background: false,
                        collectionLayout: 'dropdown'
                    }}
                ],

                // Custom rendering for badges
                columnDefs: [
                    {{
                        targets: 0, // Actions column
                        orderable: false,
                        searchable: false
                    }},
                    {{
                        targets: 4, // Category column
                        render: function(data, type, row) {{
                            if (type === 'display') {{
                                const categoryClass = 'category-' + data.toLowerCase();
                                return '<span class="badge ' + categoryClass + '">' +
                                       data + '</span>';
                            }}
                            return data;
                        }},
                        searchPanes: {{
                            show: true
                        }}
                    }},
                    {{
                        targets: 6, // Hardware column
                        searchPanes: {{
                            show: true
                        }}
                    }},
                    {{
                        targets: 7, // Status column
                        render: function(data, type, row) {{
                            if (type === 'display') {{
                                const statusClass = 'status-' + data.toLowerCase()
                                    .replace(/\\s+/g, '-').replace("'", '');
                                return '<span class="status ' + statusClass + '">' +
                                       data + '</span>';
                            }}
                            return data;
                        }},
                        searchPanes: {{
                            show: true
                        }}
                    }},
                    {{
                        targets: 8, // Priority column
                        render: function(data, type, row) {{
                            if (type === 'display') {{
                                return '<span class="badge priority-' + data + '">' +
                                       data + '</span>';
                            }}
                            return data;
                        }},
                        type: 'num',
                        searchPanes: {{
                            show: true
                        }}
                    }}
                ],

                // State saving
                stateSave: true,
                stateDuration: 60 * 60 * 24 * 7, // 1 week

                // Language customization
                language: {{
                    search: "Global Search:",
                    searchPlaceholder: "Search all columns...",
                    lengthMenu: "Show _MENU_ tests per page",
                    info: "Showing _START_ to _END_ of _TOTAL_ skipped tests",
                    infoEmpty: "No skipped tests found",
                    infoFiltered: "(filtered from _MAX_ total tests)",
                    paginate: {{
                        first: "First",
                        last: "Last",
                        next: "Next",
                        previous: "Previous"
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""

TABLE_ROW_TEMPLATE = """
            <tr>
                <td>
                    <button class="github-link"
                            onclick="window.open('{github_url}', '_blank')">View</button>
                </td>
                <td>{id}</td>
                <td>{test_name}</td>
                <td title="{skip_reason}">{skip_reason_short}</td>
                <td>{category}</td>
                <td>{dependencies}</td>
                <td>{hardware}</td>
                <td>{status}</td>
                <td>{priority}</td>
                <td>{test_location}</td>
            </tr>
"""

MODULE_CARD_TEMPLATE = """
<div class="list-group-item module-stat-item border-{color_class} py-3">
    <div class="d-flex justify-content-between align-items-center">
        <div>
            <h6 class="mb-1 fw-semibold">{module_name}</h6>
            <div class="d-flex flex-wrap gap-2 mt-2">
                <span class="badge bg-primary">Total: {total}</span>
                <span class="badge bg-success">Fixed: {fixed} ({fixed_pct}%)</span>
                <span class="badge bg-info">In Progress: {in_progress}</span>
                <span class="badge bg-warning">To Fix: {to_fix}</span>
            </div>
        </div>
        <div class="text-end">
            <div class="progress" style="width: 100px; height: 6px;">
                <div class="progress-bar bg-success" style="width: {fixed_pct}%"></div>
            </div>
            <small class="text-muted">{fixed_pct}% complete</small>
        </div>
    </div>
</div>
"""

INVENTORY_HEADER = """# Datarax Skipped Tests Inventory

> **Auto-generated**: This file is automatically maintained by
> `tests/scripts/skipped_tests_tracker.py`
> **Last Updated**: {timestamp}

## Summary

- **Total Skipped Tests**: {total_count}
- **Fixed**: {fixed_count} ({fixed_pct}%)
- **In Progress**: {in_progress_count}
- **To Fix**: {to_fix_count}
- **Won't Fix**: {wont_fix_count}

## Categories

- **HW**: Hardware-related (GPU/TPU requirements)
- **DEP**: External dependency issues
- **ISS**: Known implementation issues
- **FLAKY**: Intermittent/flaky tests
- **PERF**: Performance-related skips
- **ENV**: Environment-specific issues
- **SEL**: Selection/filtering issues
- **SEG**: Segmentation/memory issues

## Test Inventory

| **ID** | **Test Location** | **Skip Reason** | **Category** |
| **Dependencies** | **Hardware** | **Status** | **Priority** |
|--------|-------------------|-----------------|--------------|------------------|--------------|------------|--------------|
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate skipped tests tracking dashboard")
    parser.add_argument(
        "--inventory",
        type=str,
        default="docs/skipped_tests_inventory.md",
        help="Path to the skipped tests inventory Markdown file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="temp/skipped_tests_dashboard.html",
        help="Output path for the generated dashboard HTML file",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Also output results as JSON file for CI/CD integration",
    )
    parser.add_argument(
        "--scan", action="store_true", help="Scan codebase for skipped tests and update inventory"
    )
    parser.add_argument(
        "--update-status",
        nargs=2,
        metavar=("TEST_ID", "STATUS"),
        help="Update the status of a specific test",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="temp/skipped_tests_history.json",
        help="Path to history file for tracking changes",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def extract_test_name(file_path: Path, line_number: int) -> str:
    """Extract the test function or class name from a Python file."""
    try:
        with open(file_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)

        # Find the test function/class at or before the given line
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                if hasattr(node, "lineno") and node.lineno <= line_number:
                    # Check if this is a test function/class
                    if node.name.startswith("test_") or node.name.startswith("Test"):
                        # Find the closest one to our line
                        end_line = getattr(node, "end_lineno", node.lineno + 100)
                        if node.lineno <= line_number <= end_line:
                            return node.name

        # Fallback: look for test name in the vicinity
        lines = source.split("\n")
        for i in range(max(0, line_number - 10), min(len(lines), line_number + 5)):
            if "def test_" in lines[i] or "class Test" in lines[i]:
                match = re.search(r"(test_\w+|Test\w+)", lines[i])
                if match:
                    return match.group(1)

    except Exception as e:
        logger.debug(f"Failed to extract test name from {file_path}:{line_number}: {e}")

    return "unknown"


def categorize_skip_reason(reason: str) -> tuple[str, int]:
    """Intelligently categorize skip reasons and assign priority."""
    reason_lower = reason.lower()

    # Category patterns with priorities
    patterns = {
        "HW": (["gpu", "cuda", "tpu", "device", "hardware", "accelerator"], 3),
        "DEP": (["dependency", "import", "module", "package", "library", "not installed"], 2),
        "ISS": (["bug", "issue", "broken", "error", "fail", "problem", "todo", "fixme"], 1),
        "FLAKY": (["flaky", "intermittent", "unstable", "random", "sometimes"], 2),
        "PERF": (["slow", "performance", "timeout", "expensive", "memory", "oom"], 3),
        "ENV": (["environment", "ci", "github", "docker", "platform", "os"], 4),
        "SEL": (["select", "filter", "subset", "sample"], 4),
        "SEG": (["segment", "fault", "crash", "core dump"], 1),
    }

    for category, (keywords, priority) in patterns.items():
        if any(keyword in reason_lower for keyword in keywords):
            return category, priority

    # Default category and priority
    return "ISS", 3


def find_skipped_tests() -> list[dict[str, Any]]:
    """Scan the codebase for skipped tests with enhanced metadata extraction."""
    logger.info("Scanning codebase for skipped tests...")

    skip_patterns = [
        (r"pytest\.skip\(['\"](.+?)['\"]\)", "skip"),
        (r"@pytest\.mark\.skip\(reason=['\"](.+?)['\"]\)", "mark.skip"),
        (r"@pytest\.mark\.skipif\((.+?),\s*reason=['\"](.+?)['\"]\)", "mark.skipif"),
        (r"@pytest\.mark\.xfail\(.*?reason=['\"](.+?)['\"]\)", "mark.xfail"),
    ]

    tests_dir = Path("tests")
    skipped_tests = []

    for file_path in tests_dir.glob("**/*.py"):
        relative_path = file_path.relative_to(Path("."))
        module_name = str(relative_path.parent).replace("/", ".").replace("tests.", "")

        with open(file_path, "r") as f:
            try:
                content = f.read()
                for i, line in enumerate(content.splitlines(), 1):
                    for pattern, skip_type in skip_patterns:
                        matches = re.search(pattern, line)
                        if matches:
                            if skip_type == "mark.skipif":
                                condition = matches.group(1)
                                reason = matches.group(2)
                            else:
                                reason = matches.group(1)
                                condition = None

                            test_name = extract_test_name(file_path, i)
                            category, priority = categorize_skip_reason(reason)

                            # Generate unique ID based on file and line
                            test_id = (
                                hashlib.md5(f"{relative_path}:{i}".encode()).hexdigest()[:8].upper()
                            )

                            skipped_tests.append(
                                {
                                    "id": test_id,
                                    "file": str(relative_path),
                                    "line": i,
                                    "test_name": test_name,
                                    "reason": reason,
                                    "full_line": line.strip(),
                                    "skip_type": skip_type,
                                    "condition": condition,
                                    "module": module_name,
                                    "category": category,
                                    "priority": priority,
                                    "status": "To Fix",  # Default status for new tests
                                    "dependencies": "",
                                    "hardware": "GPU"
                                    if "gpu" in reason.lower() or "cuda" in reason.lower()
                                    else "Any",
                                }
                            )
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode file: {file_path}")

    logger.info(f"Found {len(skipped_tests)} skipped tests")
    return skipped_tests


def parse_inventory_file(inventory_path: str) -> list[dict[str, Any]]:
    """Parse the inventory Markdown file to extract test information."""
    if not os.path.exists(inventory_path):
        logger.warning(f"Inventory file not found: {inventory_path}")
        return []

    logger.info(f"Parsing inventory file: {inventory_path}")

    with open(inventory_path, "r") as f:
        content = f.read()

    # Extract table rows - now with test name field
    table_pattern = (
        r"\| \*\*([A-Z0-9]+)\*\* \| (.+?) \| (.+?) \| ([A-Z]+) \| "
        r"(.*?) \| (.+?) \| (.+?) \| (\d+) \|"
    )
    matches = re.findall(table_pattern, content)

    inventory = []
    for match in matches:
        test_id, test_location, skip_reason, category, dependencies, hardware, status, priority = (
            match
        )

        # Extract test name from location if available
        test_name = "unknown"
        if "::" in test_location:
            parts = test_location.split("::")
            if len(parts) >= 2:
                test_name = parts[-1]

        inventory.append(
            {
                "id": test_id,
                "test_location": test_location.strip(),
                "test_name": test_name,
                "skip_reason": skip_reason.strip(),
                "category": category,
                "dependencies": dependencies.strip(),
                "hardware": hardware.strip(),
                "status": status.strip(),
                "priority": int(priority),
            }
        )

    logger.info(f"Parsed {len(inventory)} tests from inventory")
    return inventory


def update_inventory_file(inventory_path: str, tests: list[dict[str, Any]]) -> None:
    """Update or create the inventory Markdown file."""
    logger.info(f"Updating inventory file: {inventory_path}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(inventory_path) or ".", exist_ok=True)

    # Sort tests by module and then by ID
    tests = sorted(tests, key=lambda x: (x.get("module", ""), x["id"]))

    # Calculate statistics
    total_count = len(tests)
    fixed_count = sum(1 for t in tests if t.get("status") == "Fixed")
    in_progress_count = sum(1 for t in tests if t.get("status") == "In Progress")
    to_fix_count = sum(1 for t in tests if t.get("status") == "To Fix")
    wont_fix_count = sum(1 for t in tests if t.get("status") == "Won't Fix")
    fixed_pct = round(fixed_count / total_count * 100) if total_count > 0 else 0

    # Build the markdown content
    content = INVENTORY_HEADER.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_count=total_count,
        fixed_count=fixed_count,
        fixed_pct=fixed_pct,
        in_progress_count=in_progress_count,
        to_fix_count=to_fix_count,
        wont_fix_count=wont_fix_count,
    )

    # Add test entries
    for test in tests:
        # Build test location with test name
        test_location = test.get("file", test.get("test_location", ""))
        if test.get("line"):
            test_location += f":{test['line']}"
        if test.get("test_name") and test["test_name"] != "unknown":
            test_location += f"::{test['test_name']}"

        # Truncate long reasons for the table
        reason = test.get("reason", test.get("skip_reason", ""))
        if len(reason) > 60:
            reason = reason[:57] + "..."

        content += (
            f"| **{test['id']}** | {test_location} | {reason} | {test.get('category', 'ISS')} | "
        )
        content += f"{test.get('dependencies', '')} | {test.get('hardware', 'Any')} | "
        content += f"{test.get('status', 'To Fix')} | {test.get('priority', 3)} |\n"

    # Write the file
    with open(inventory_path, "w") as f:
        f.write(content)

    logger.info(f"Inventory file updated with {total_count} tests")


def merge_with_inventory(
    scanned_tests: list[dict[str, Any]], inventory: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Merge newly scanned tests with existing inventory."""
    # Create lookup maps
    inventory_map = {item["id"]: item for item in inventory}

    # Also create a map by location for matching
    location_map = {}
    for item in inventory:
        loc_key = item["test_location"].split("::")[0]  # Get file:line part
        location_map[loc_key] = item

    merged = []
    new_count = 0

    for test in scanned_tests:
        test_location = f"{test['file']}:{test['line']}"

        # Check if test exists in inventory by ID or location
        existing = None
        if test["id"] in inventory_map:
            existing = inventory_map[test["id"]]
        elif test_location in location_map:
            existing = location_map[test_location]

        if existing:
            # Merge with existing, preserving manual updates
            merged_test = {
                **test,
                "status": existing["status"],
                "dependencies": existing.get("dependencies", ""),
                "priority": existing.get("priority", test["priority"]),
                # Update category if reason changed significantly
                "category": test["category"]
                if test["reason"] != existing.get("skip_reason")
                else existing["category"],
            }
        else:
            # New test found
            merged_test = test
            new_count += 1
            logger.info(f"New skipped test found: {test_location} - {test['reason'][:50]}...")

        merged.append(merged_test)

    return merged, new_count


def track_history(history_path: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Track changes over time."""
    history = []

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

    # Add current stats
    stats["timestamp"] = datetime.now().isoformat()
    history.append(stats)

    # Keep only last 100 entries
    history = history[-100:]

    # Save updated history
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Calculate trends
    trends = {}
    if len(history) > 1:
        prev = history[-2]
        curr = history[-1]
        trends = {
            "total_change": curr.get("total_count", 0) - prev.get("total_count", 0),
            "fixed_change": curr.get("fixed_count", 0) - prev.get("fixed_count", 0),
            "progress_change": curr.get("progress", 0) - prev.get("progress", 0),
        }

    return trends


def generate_module_stats(tests: list[dict[str, Any]]) -> str:
    """Generate module statistics HTML."""
    module_stats = defaultdict(lambda: {"total": 0, "fixed": 0, "in_progress": 0, "to_fix": 0})

    for test in tests:
        module = test.get("module", "unknown")
        module_stats[module]["total"] += 1

        status = test.get("status", "To Fix")
        if status == "Fixed":
            module_stats[module]["fixed"] += 1
        elif status == "In Progress":
            module_stats[module]["in_progress"] += 1
        elif status == "To Fix":
            module_stats[module]["to_fix"] += 1

    # Skip if no modules to show
    if not module_stats:
        return ""

    # Wrap in a modern card with list group
    html = """
    <div class="card">
        <div class="card-header">
            <h5 class="mb-0">Module Statistics</h5>
        </div>
        <div class="list-group list-group-flush">
    """

    # Sort modules and add each as a list item
    for i, (module, stats) in enumerate(sorted(module_stats.items())):
        fixed_pct = round(stats["fixed"] / stats["total"] * 100) if stats["total"] > 0 else 0

        # Determine color class based on completion percentage
        if fixed_pct >= 80:
            color_class = "success"
        elif fixed_pct >= 50:
            color_class = "info"
        elif fixed_pct >= 20:
            color_class = "warning"
        else:
            color_class = "danger"

        html += MODULE_CARD_TEMPLATE.format(
            module_name=module or "root",
            total=stats["total"],
            fixed=stats["fixed"],
            fixed_pct=fixed_pct,
            in_progress=stats["in_progress"],
            to_fix=stats["to_fix"],
            color_class=color_class,
        )

    html += """
        </div>
    </div>
    """

    return html


def get_github_url(test_location: str) -> str:
    """Generate a GitHub URL for a test file."""
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
            parts = test_location.split(":")
            file_path = parts[0]
            line_num = ""
            if len(parts) > 1 and parts[1].isdigit():
                line_num = f"L{parts[1]}"

            return f"https://github.com/{owner_repo}/blob/{branch}/{file_path}#{line_num}"
    except Exception:
        pass

    return "#"


def generate_dashboard(
    inventory: list[dict[str, Any]], output_path: str, new_count: int = 0
) -> dict[str, Any]:
    """Generate HTML dashboard from inventory data."""
    logger.info(f"Generating dashboard at: {output_path}")

    # Compute statistics
    total_count = len(inventory)
    fixed_count = sum(1 for item in inventory if item.get("status") == "Fixed")
    in_progress_count = sum(1 for item in inventory if item.get("status") == "In Progress")
    to_fix_count = sum(1 for item in inventory if item.get("status") == "To Fix")
    wont_fix_count = sum(1 for item in inventory if item.get("status") == "Won't Fix")

    progress = round(fixed_count / total_count * 100) if total_count > 0 else 0

    # Generate module statistics
    module_stats_html = generate_module_stats(inventory)

    # Generate table rows
    table_rows = ""
    for item in inventory:
        github_url = get_github_url(item.get("test_location", ""))

        # Shorten reason for display
        reason = item.get("skip_reason", item.get("reason", ""))
        reason_short = reason[:60] + "..." if len(reason) > 60 else reason

        table_rows += TABLE_ROW_TEMPLATE.format(
            id=item["id"],
            test_location=item.get("test_location", item.get("file", "")),
            test_name=item.get("test_name", "unknown"),
            skip_reason=reason,
            skip_reason_short=reason_short,
            category=item.get("category", "ISS"),
            priority=item.get("priority", 3),
            status=item.get("status", "To Fix"),
            dependencies=item.get("dependencies", ""),
            hardware=item.get("hardware", "Any"),
            github_url=github_url,
        )

    # Generate HTML content
    html_content = REPORT_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        progress=progress,
        total_count=total_count,
        fixed_count=fixed_count,
        in_progress_count=in_progress_count,
        to_fix_count=to_fix_count,
        wont_fix_count=wont_fix_count,
        new_count=new_count,
        module_stats=module_stats_html,
        table_rows=table_rows,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write HTML file
    with open(output_path, "w") as f:
        f.write(html_content)

    logger.info(f"Dashboard generated successfully at: {output_path}")

    # Return statistics for other uses
    return {
        "total_count": total_count,
        "fixed_count": fixed_count,
        "in_progress_count": in_progress_count,
        "to_fix_count": to_fix_count,
        "wont_fix_count": wont_fix_count,
        "progress": progress,
        "new_count": new_count,
    }


def generate_json_output(
    inventory: list[dict[str, Any]], output_path: str, stats: dict[str, Any]
) -> None:
    """Generate JSON output for CI/CD integration."""
    logger.info(f"Generating JSON output at: {output_path}")

    # Prepare JSON data
    json_data = {
        "generated_at": datetime.now().isoformat(),
        "statistics": stats,
        "tests": inventory,
        "summary": {
            "total": stats["total_count"],
            "by_status": {
                "fixed": stats["fixed_count"],
                "in_progress": stats["in_progress_count"],
                "to_fix": stats["to_fix_count"],
                "wont_fix": stats["wont_fix_count"],
            },
            "by_category": defaultdict(int),
            "by_priority": defaultdict(int),
            "by_module": defaultdict(int),
        },
    }

    # Aggregate by different dimensions
    for test in inventory:
        json_data["summary"]["by_category"][test.get("category", "ISS")] += 1
        json_data["summary"]["by_priority"][test.get("priority", 3)] += 1
        json_data["summary"]["by_module"][test.get("module", "unknown")] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    json_data["summary"]["by_category"] = dict(json_data["summary"]["by_category"])
    json_data["summary"]["by_priority"] = dict(json_data["summary"]["by_priority"])
    json_data["summary"]["by_module"] = dict(json_data["summary"]["by_module"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write JSON file
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info("JSON output generated successfully")


def update_test_status(inventory_path: str, test_id: str, new_status: str) -> None:
    """Update the status of a specific test in the inventory."""
    valid_statuses = ["Fixed", "In Progress", "To Fix", "Won't Fix"]
    if new_status not in valid_statuses:
        logger.error(f"Invalid status '{new_status}'. Must be one of: {', '.join(valid_statuses)}")
        return

    inventory = parse_inventory_file(inventory_path)

    found = False
    for test in inventory:
        if test["id"] == test_id.upper():
            old_status = test["status"]
            test["status"] = new_status
            logger.info(f"Updated test {test_id}: {old_status} -> {new_status}")
            found = True
            break

    if not found:
        logger.error(f"Test ID '{test_id}' not found in inventory")
        return

    # Update the inventory file
    update_inventory_file(inventory_path, inventory)


def main() -> None:
    """Execute the main program logic."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle status update command
    if args.update_status:
        update_test_status(args.inventory, args.update_status[0], args.update_status[1])
        return

    inventory = []
    new_count = 0

    if args.scan:
        # Scan codebase for skipped tests
        scanned_tests = find_skipped_tests()

        # Load existing inventory if it exists
        existing_inventory = parse_inventory_file(args.inventory)

        # Merge with existing inventory
        inventory, new_count = merge_with_inventory(scanned_tests, existing_inventory)

        # Update the inventory file
        update_inventory_file(args.inventory, inventory)

        if new_count > 0:
            logger.info(f"Found {new_count} new skipped tests")
    else:
        # Just load from existing inventory
        if not os.path.exists(args.inventory):
            logger.error(f"Inventory file not found: {args.inventory}")
            logger.info("Run with --scan flag to create initial inventory")
            return

        inventory = parse_inventory_file(args.inventory)

    # Generate dashboard
    stats = generate_dashboard(inventory, args.output, new_count)

    # Track history
    trends = track_history(args.history, stats)
    if trends:
        logger.info(
            f"Trends: Total {trends['total_change']:+d}, "
            f"Fixed {trends['fixed_change']:+d}, "
            f"Progress {trends['progress_change']:+.1f}%"
        )

    # Generate JSON output if requested
    if args.json:
        generate_json_output(inventory, args.json, stats)

    # Print summary
    logger.info("=" * 60)
    logger.info(
        f"Summary: {stats['total_count']} total, "
        f"{stats['fixed_count']} fixed ({stats['progress']}%)"
    )
    logger.info(
        f"Status: {stats['in_progress_count']} in progress, "
        f"{stats['to_fix_count']} to fix, {stats['wont_fix_count']} won't fix"
    )
    if new_count > 0:
        logger.info(f"New tests found: {new_count}")
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
