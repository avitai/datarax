"""Tests for scripts/prepare_example_datasets.py, which fills CI's cache of example datasets."""

from __future__ import annotations

import re
from types import ModuleType

import pytest

from tests.scripts.script_loader import load_script, REPO_ROOT


_TFDS_CIFAR10 = re.compile(r'Config\(\s*name="cifar10"|tfds\.(?:load|builder)\("cifar10"')
_KERAS_CIFAR10 = re.compile(r"from keras\.datasets import cifar10\b")


@pytest.fixture(scope="module")
def prepare() -> ModuleType:
    return load_script("prepare_example_datasets")


def _cifar10_loaders() -> set[tuple[str, str]]:
    """The CIFAR-10 loaders the examples call, found in their sources."""
    loaders: set[tuple[str, str]] = set()
    for path in sorted((REPO_ROOT / "examples").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if _TFDS_CIFAR10.search(text):
            loaders.add(("tfds", "cifar10"))
        if _KERAS_CIFAR10.search(text):
            loaders.add(("keras", "cifar10"))
    return loaders


def test_every_cifar10_loader_the_examples_use_is_prepared(prepare: ModuleType) -> None:
    """CIFAR-10 is served from a slow host, so each loader of it an example uses is cached in CI."""
    loaders = _cifar10_loaders()
    prepared = {("tfds", name) for name in prepare.TFDS_DATASETS} | {
        ("keras", name) for name in prepare.KERAS_DATASETS
    }

    assert loaders == {("tfds", "cifar10"), ("keras", "cifar10")}
    assert loaders <= prepared


def test_prepare_all_runs_each_dataset_loader_once(
    prepare: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(prepare, "prepare_tfds_dataset", lambda name: calls.append(("tfds", name)))
    monkeypatch.setattr(
        prepare, "prepare_keras_dataset", lambda name: calls.append(("keras", name))
    )

    prepare.prepare_all()

    expected = [("tfds", name) for name in prepare.TFDS_DATASETS]
    expected += [("keras", name) for name in prepare.KERAS_DATASETS]
    assert sorted(calls) == sorted(expected)
