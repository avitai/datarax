# source_ops

The operations a data source is built from. `MemorySource`, the HuggingFace and TFDS
sources, and sources in other packages (DiffAV's WOD source, DiffBio's AnnData source)
compose these instead of re-implementing index resolution, iteration, batching and reset.

::: datarax.sources.source_ops
