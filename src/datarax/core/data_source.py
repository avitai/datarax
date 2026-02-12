"""Base module for data sources in Datarax.

This module defines the base class for all Datarax data source components
that use flax.nnx.Module for state management and JAX transformation
compatibility.
"""

from collections.abc import Iterator


from datarax.core.structural import StructuralModule
from datarax.typing import Element


class DataSourceModule(StructuralModule):
    """Enhanced base module for all Datarax data source components.

    This class extends StructuralModule for non-parametric structural data loading.
    Concrete data sources define their own config classes extending StructuralConfig.

    A DataSourceModule is responsible for reading data from an external source
    (e.g., files, memory, network) and yielding data elements as PyTrees. Each
    data element is typically a dictionary or other PyTree structure containing
    JAX arrays or Python primitives.

    **Important**: When subclassing, if you store data containing JAX Arrays in
    an attribute (like `self.data`), you MUST annotate it with `nnx.Param` or `nnx.data()`:

    Examples:
        ```python
        @dataclass
        class MyDataSourceConfig(StructuralConfig):
            required_param: int | None = None
            def __post_init__(self):
                super().__post_init__()
                if self.required_param is None:
                    raise ValueError("required_param is required")
        class MyDataSource(DataSourceModule):
            data: list[dict] = nnx.data()  # REQUIRED: Annotate data attribute with nnx.data()
            def __init__(self, config: MyDataSourceConfig, data: list[dict], *,
                         rngs: nnx.Rngs | None = None, name: str | None = None):
                super().__init__(config, rngs=rngs, name=name)
                self.data = data  # Safe with nnx.Param/nnx.data() annotation
        ```

    This prevents NNX from trying to track individual JAX Arrays within the data
    structure as trainable parameters.
    """

    def __iter__(self) -> Iterator[Element]:
        """Return an iterator over individual data elements.

        Returns:
            An iterator that yields data elements as PyTrees.
        """
        raise NotImplementedError("Subclasses must implement __iter__")

    def __next__(self) -> Element:
        """Get the next element from this data source.

        Returns:
            The next data element as a PyTree.

        Raises:
            StopIteration: When there are no more elements to yield.
        """
        raise NotImplementedError("Subclasses must implement __next__")

    def __len__(self) -> int:
        """Return the total number of data elements.

        Implementing this method allows downstream components to know the
        dataset size in advance, which can be useful for progress tracking
        or specific sampling strategies.

        Returns:
            The total number of data elements in the source.

        Raises:
            NotImplementedError: If the source cannot determine its length.
        """
        raise NotImplementedError("This DataSourceModule does not support length determination.")

    def __getitem__(self, idx: int) -> Element | None:
        """Get element by index.

        This method provides subscriptable access to data elements.
        Subclasses should override this method if they support random access.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The data element at the given index, or None if not implemented.
        """
        return None
