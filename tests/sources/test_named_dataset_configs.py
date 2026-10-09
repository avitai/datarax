"""The configs of sources reading a named dataset validate once, in their common base.

``HFEagerConfig``, ``HFStreamingConfig``, ``TFDSEagerConfig`` and ``TFDSStreamingConfig`` share
their dataset fields (name, split, key filters) and the two stream configs their shuffle buffer.
Each rule is checked by the base that declares the field, through the ordinary
``super().__post_init__()`` chain, so a subclass of any of them constructs, is validated as its
parent is, and may add a ``__post_init__`` of its own that calls ``super()``.
"""

from __future__ import annotations

from dataclasses import dataclass, FrozenInstanceError

import pytest

from datarax.sources import (
    HFEagerConfig,
    HFStreamingConfig,
    TFDSEagerConfig,
    TFDSStreamingConfig,
)


_CONFIGS = [HFEagerConfig, HFStreamingConfig, TFDSEagerConfig, TFDSStreamingConfig]
_STREAM_CONFIGS = [HFStreamingConfig, TFDSStreamingConfig]


def _subclass(parent: type) -> type:
    @dataclass(frozen=True)
    class Sub(parent):
        extra: int = 1

    return Sub


def _subclass_checking_its_own_field(parent: type) -> type:
    @dataclass(frozen=True)
    class Checked(parent):
        extra: int = 1

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.extra < 0:
                raise ValueError(f"extra must not be negative; got {self.extra}")

    return Checked


@pytest.mark.parametrize("parent", _CONFIGS, ids=lambda config: config.__name__)
class TestASubclass:
    def test_constructs_with_its_parent_s_fields_and_its_own(self, parent: type) -> None:
        config = _subclass(parent)(name="mnist", split="train", extra=3)

        assert (config.name, config.split, config.extra) == ("mnist", "train", 3)

    def test_is_validated_as_its_parent_naming_itself(self, parent: type) -> None:
        sub = _subclass(parent)

        with pytest.raises(ValueError, match="name is required for Sub"):
            sub(split="train")
        with pytest.raises(ValueError, match="split is required for Sub"):
            sub(name="mnist")
        with pytest.raises(ValueError, match="Cannot specify both"):
            sub(name="mnist", split="train", include_keys={"a"}, exclude_keys={"b"})

    def test_overriding_post_init_and_calling_super_runs_both_checks(self, parent: type) -> None:
        checked = _subclass_checking_its_own_field(parent)

        assert checked(name="mnist", split="train").extra == 1
        with pytest.raises(ValueError, match="extra must not be negative"):
            checked(name="mnist", split="train", extra=-1)
        with pytest.raises(ValueError, match="name is required for Checked"):
            checked(split="train")


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda config: config.__name__)
class TestTheBaseRules:
    def test_a_config_names_itself_in_a_refusal(self, config: type) -> None:
        with pytest.raises(ValueError, match=f"name is required for {config.__name__}"):
            config(split="train")
        with pytest.raises(ValueError, match=f"split is required for {config.__name__}"):
            config(name="mnist")

    def test_key_filters_given_as_sets_are_held_immutable_and_the_config_hashes(
        self, config: type
    ) -> None:
        built = config(name="mnist", split="train", include_keys={"image", "label"})

        assert built.include_keys == frozenset({"image", "label"})
        assert isinstance(built.include_keys, frozenset)
        held = config(name="mnist", split="train", include_keys=frozenset({"label", "image"}))
        assert built == held
        assert hash(built) == hash(held)
        with pytest.raises(FrozenInstanceError):
            built.include_keys = None  # type: ignore[misc]


@pytest.mark.parametrize("config", _STREAM_CONFIGS, ids=lambda config: config.__name__)
def test_a_stream_config_and_its_subclass_refuse_an_empty_shuffle_buffer(config: type) -> None:
    for built in (config, _subclass(config)):
        assert built(name="mnist", split="train").shuffle_buffer_size == 1000
        with pytest.raises(ValueError, match="shuffle_buffer_size must be at least 1"):
            built(name="mnist", split="train", shuffle_buffer_size=0)
