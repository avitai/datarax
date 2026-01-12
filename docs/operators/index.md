# Operators

This section contains documentation for data transformation operators.

## Core Operators

Base operator types for building data pipelines:

- [batch_mix_operator](batch_mix_operator.md) - Batch mixing operations
- [composite_operator](composite_operator.md) - Compose multiple operators
- [element_operator](element_operator.md) - Element-wise operations
- [map_operator](map_operator.md) - Map transformations
- [probabilistic_operator](probabilistic_operator.md) - Probabilistic application
- [selector_operator](selector_operator.md) - Random operator selection

## Image Operators

Specialized operators for image data transformations:

- [brightness_operator](brightness_operator.md) - Brightness adjustments
- [contrast_operator](contrast_operator.md) - Contrast modifications
- [dropout_operator](dropout_operator.md) - Pixel dropout
- [noise_operator](noise_operator.md) - Noise injection
- [patch_dropout_operator](patch_dropout_operator.md) - Patch-level dropout
- [rotation_operator](rotation_operator.md) - Image rotation
- [functional](functional.md) - Functional image operations

## Strategies

Operator composition and execution strategies:

- [base](base.md) - Base strategy interface
- [branching](branching.md) - Conditional branching
- [ensemble](ensemble.md) - Ensemble execution
- [merging](merging.md) - Output merging
- [parallel](parallel.md) - Parallel execution
- [sequential](sequential.md) - Sequential execution
