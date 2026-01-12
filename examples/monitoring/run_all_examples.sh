#!/bin/bash

# Script to run all Datarax monitoring examples

# Ensure we're in the repository root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$(dirname "$SCRIPT_DIR")"/../ || exit 1

echo "================================================================"
echo "Running Datarax Monitoring Examples"
echo "================================================================"
echo ""

# Run all examples
echo "Running simple monitoring example..."
python examples/monitoring/simple_monitoring.py

echo ""
echo "================================================================"
echo ""

echo "Running file reporter example..."
python examples/monitoring/file_reporter_example.py

echo ""
echo "================================================================"
echo "All monitoring examples completed"
echo "================================================================"
