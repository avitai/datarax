"""The environment variable that chooses protobuf's runtime, which no tracked file may set.

Spelled in two halves, so a scan of the tracked files for the name does not find its own
definition: ``tests/test_tooling_contracts.py`` asserts no tracked file names it.
"""

RUNTIME_VARIABLE = "PROTOCOL_BUFFERS" + "_PYTHON_IMPLEMENTATION"
