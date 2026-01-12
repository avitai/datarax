# DAG

Directed Acyclic Graph (DAG) based pipeline execution system.

## Core Modules

- [dag_config](dag_config.md) - DAG configuration and setup
- [dag_executor](dag_executor.md) - DAG execution engine

## Node Types

Nodes are the building blocks of DAG pipelines:

- [base](base.md) - Base node classes and interfaces
- [caching](caching.md) - Caching nodes for intermediate results
- [control_flow](control_flow.md) - Conditional execution and branching
- [data_source](data_source.md) - Data source nodes
- [field_operators](field_operators.md) - Field-level transformation nodes
- [loaders](loaders.md) - Data loading nodes
- [rebatch](rebatch.md) - Batch reshaping nodes
