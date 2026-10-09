"""The configuration bases of sources reading a named dataset (TFDS, HuggingFace)."""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from datarax.core.config import StructuralConfig


@dataclass(frozen=True)
class SourceConfigBase(StructuralConfig):
    """The fields every named-dataset source config holds, validated where they are declared.

    A subclass is validated as this base is, through ``super().__post_init__()``; refusals name
    the class constructed. The key filters are held as ``frozenset``s, so a configuration is
    hashable and compares equal whichever set type it was given.

    Attributes:
        name: The dataset's name (required).
        split: The split to read (required).
        data_dir: Where the backend finds the dataset; its meaning is the backend's.
        include_keys: The only keys to keep (exclusive with ``exclude_keys``).
        exclude_keys: Keys to drop (exclusive with ``include_keys``).
    """

    name: str | None = None
    split: str | None = None
    data_dir: str | None = None
    include_keys: AbstractSet[str] | None = None
    exclude_keys: AbstractSet[str] | None = None

    def __post_init__(self) -> None:
        """Validate the dataset fields and hold the key filters immutable.

        Raises:
            ValueError: If the name or the split is missing, or both key filters are given.
        """
        super().__post_init__()
        config = type(self).__name__
        if self.name is None:
            raise ValueError(f"name is required for {config}")
        if self.split is None:
            raise ValueError(f"split is required for {config}")
        if self.include_keys is not None and self.exclude_keys is not None:
            raise ValueError("Cannot specify both include_keys and exclude_keys")
        for field in ("include_keys", "exclude_keys"):
            keys = getattr(self, field)
            if keys is not None:
                object.__setattr__(self, field, frozenset(keys))


@dataclass(frozen=True)
class StreamingSourceConfigBase(SourceConfigBase):
    """A named-dataset stream's config: the dataset fields and the shuffle buffer.

    Attributes:
        shuffle_buffer_size: Records the shuffle buffer holds when the pipeline shuffles.
    """

    shuffle_buffer_size: int = 1000

    def __post_init__(self) -> None:
        """Validate the dataset fields and the buffer.

        Raises:
            ValueError: If a dataset field is invalid (see :class:`SourceConfigBase`) or the
                buffer holds no record.
        """
        super().__post_init__()
        if self.shuffle_buffer_size < 1:
            raise ValueError(
                f"shuffle_buffer_size must be at least 1; got {self.shuffle_buffer_size}"
            )
