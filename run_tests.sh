#!/bin/bash
# Thin wrapper — delegates to the main test runner in scripts/
exec "$(dirname "$0")/scripts/run_tests.sh" "$@"
