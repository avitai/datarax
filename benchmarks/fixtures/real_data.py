"""Framework-neutral real-data provider for benchmark scenarios.

Mirrors the :class:`~benchmarks.fixtures.synthetic_data.SyntheticDataGenerator`
output contract: every method returns raw numpy arrays, so each adapter wraps
byte-identical input its own way (datarax -> MemorySource, PyTorch ->
``from_numpy``, ...). Nothing here hands out framework-specific objects; that
is what keeps cross-framework comparisons fair.

Datasets (cached locally after a one-time materialization):

- cifar10 via ``tensorflow_datasets`` (honors ``TFDS_DATA_DIR``)
- wikitext-103-raw-v1 via Hugging Face ``Salesforce/wikitext``
- Criteo DAC sample (13 dense + 26 categorical) via ``Recommenders/criteo``
- COCO captions (Karpathy validation split) via ``jxie/coco_captions``

Downloads are disabled by default; set ``DATARAX_BENCH_DOWNLOAD=1`` (or pass
``allow_download=True``) to permit them. With downloads disabled, missing
data raises :class:`RealDataUnavailableError` with remediation instructions.

Selection is deterministic per seed: rows are permuted once, then tiled
cyclically when a request exceeds the dataset size.
"""

from __future__ import annotations

import io
import os
import tarfile
import zlib
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DOWNLOAD_ENV_VAR: str = "DATARAX_BENCH_DOWNLOAD"
"""Environment variable that enables dataset downloads when set to ``"1"``."""

DEFAULT_SEED: int = 42
"""Default RNG seed, matching the synthetic generator."""

_DAC_NUM_DENSE: int = 13
_DAC_NUM_SPARSE: int = 26
_DAC_NUM_FIELDS: int = 1 + _DAC_NUM_DENSE + _DAC_NUM_SPARSE

# Hugging Face repos pinned to commit SHAs: reproducible bytes across
# machines and runs, immune to upstream re-uploads.
_WIKITEXT_REPO: str = "Salesforce/wikitext"
_WIKITEXT_CONFIG: str = "wikitext-103-raw-v1"
_WIKITEXT_REVISION: str = "b08601e04326c79dfdd32d625aee71d232d685c3"
_CRITEO_REPO: str = "Recommenders/criteo"
_CRITEO_ARCHIVE: str = "dac_sample.tar.gz"
_CRITEO_REVISION: str = "1d462d3f6add11c4b4d5b7567da59a3311ee909e"
_COCO_REPO: str = "jxie/coco_captions"
_COCO_REVISION: str = "a2ed90d49b61dd13dd71f399c70f5feb897f8bec"


class RealDataUnavailableError(RuntimeError):
    """A real dataset is not cached locally and downloads are disabled."""


def hash_tokenize(words: Sequence[str], vocab_size: int) -> np.ndarray:
    """Map words to deterministic token ids via CRC-32 hashing.

    Word identity is preserved (same word -> same id), so real text keeps
    its Zipfian repetition structure without requiring a trained tokenizer.

    Args:
        words: Words to tokenize.
        vocab_size: Ids are in ``[0, vocab_size)``.

    Returns:
        Array of shape ``(len(words),)`` with dtype int32.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    return np.fromiter(
        (zlib.crc32(word.encode("utf-8")) % vocab_size for word in words),
        dtype=np.int32,
        count=len(words),
    )


def default_tfds_data_dir() -> Path:
    """Return the TFDS data directory (``TFDS_DATA_DIR`` or the home default)."""
    return Path(os.environ.get("TFDS_DATA_DIR", str(Path.home() / "tensorflow_datasets")))


def cifar10_is_cached(data_dir: str | Path | None = None) -> bool:
    """Return True if cifar10 is already materialized in the TFDS data dir."""
    base = Path(data_dir) if data_dir is not None else default_tfds_data_dir()
    return (base / "cifar10").is_dir()


def _download_hint(dataset: str) -> str:
    """Build the standard remediation message for missing datasets."""
    return (
        f"{dataset} is not cached locally and downloads are disabled. "
        f"Set {DOWNLOAD_ENV_VAR}=1 (or pass allow_download=True) to "
        f"materialize it once; subsequent runs use the local cache."
    )


def _require_positive(n: int) -> None:
    """Reject non-positive sample counts."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")


def _select_indices(available: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Permute ``available`` row indices once, tiling cyclically up to ``n``."""
    return np.resize(rng.permutation(available), n)


def _resize_image(image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinearly resize one HWC uint8 image to ``(h, w)``."""
    if image.shape[:2] == (h, w):
        return image
    resized = Image.fromarray(image).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _resize_images(images: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinearly resize a batch of NHWC uint8 images to ``(h, w)``."""
    if images.shape[1:3] == (h, w):
        return images
    # Preallocate: a list + np.stack would double peak memory on large batches.
    resized = np.empty((images.shape[0], h, w, images.shape[3]), dtype=np.uint8)
    for index, image in enumerate(images):
        resized[index] = _resize_image(image, h, w)
    return resized


def _caption_tokens(caption: str, text_len: int, vocab_size: int) -> np.ndarray:
    """Hash-tokenize a caption, truncating/zero-padding to ``text_len``."""
    ids = hash_tokenize(caption.split()[:text_len], vocab_size)
    padded = np.zeros(text_len, dtype=np.int32)
    padded[: len(ids)] = ids
    return padded


# ---------------------------------------------------------------------------
# Raw dataset loaders (module-level seams, replaced by fakes in unit tests)
# ---------------------------------------------------------------------------


def _load_cifar10_train(data_dir: str | Path | None, allow_download: bool) -> np.ndarray:
    """Load the full cifar10 train split as a uint8 NHWC array via TFDS."""
    if not allow_download and not cifar10_is_cached(data_dir):
        raise RealDataUnavailableError(_download_hint("cifar10 (TFDS)"))
    import tensorflow_datasets as tfds

    batch = tfds.load(
        "cifar10",
        split="train",
        data_dir=str(data_dir) if data_dir is not None else None,
        batch_size=-1,
        download=allow_download,
    )
    return np.asarray(tfds.as_numpy(batch)["image"], dtype=np.uint8)


def _load_hf_dataset(
    path: str, name: str | None, split: str, revision: str, allow_download: bool
) -> Any:
    """Load a Hugging Face dataset split, honoring the download guard."""
    import datasets as hf_datasets

    download_config = hf_datasets.DownloadConfig(local_files_only=not allow_download)
    try:
        # Revision is always a pinned commit SHA (module constants above);
        # bandit cannot resolve it through the parameter.
        return hf_datasets.load_dataset(  # nosec B615
            path, name=name, split=split, revision=revision, download_config=download_config
        )
    except (OSError, ConnectionError) as e:
        # All HF offline/missing-data failures derive from OSError
        # (LocalEntryNotFoundError, DatasetNotFoundError) or ConnectionError.
        raise RealDataUnavailableError(_download_hint(f"{path} (Hugging Face)")) from e


def _load_wikitext_lines(allow_download: bool) -> Iterable[str]:
    """Yield raw text lines from the wikitext-103 train split."""
    dataset = _load_hf_dataset(
        _WIKITEXT_REPO, _WIKITEXT_CONFIG, "train", _WIKITEXT_REVISION, allow_download
    )
    for batch in dataset.iter(batch_size=1024):
        yield from batch["text"]


def _load_criteo_lines(allow_download: bool) -> Iterable[str]:
    """Return the non-empty DAC-format lines of the Criteo sample archive."""
    from huggingface_hub import hf_hub_download

    try:
        # Pinned to a commit SHA; bandit cannot resolve the constant.
        archive_path = hf_hub_download(  # nosec B615
            repo_id=_CRITEO_REPO,
            filename=_CRITEO_ARCHIVE,
            repo_type="dataset",
            revision=_CRITEO_REVISION,
            local_files_only=not allow_download,
        )
    except OSError as e:
        # hf_hub offline/missing-file errors all derive from OSError.
        raise RealDataUnavailableError(_download_hint("Criteo DAC sample")) from e

    with tarfile.open(archive_path, "r:gz") as archive:
        member = next((m for m in archive.getmembers() if m.name.endswith(".txt")), None)
        if member is None:
            raise RealDataUnavailableError(f"{_CRITEO_ARCHIVE} contains no .txt member")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RealDataUnavailableError(f"could not read {member.name} from {_CRITEO_ARCHIVE}")
        with io.TextIOWrapper(extracted, encoding="utf-8") as text:
            return [line.rstrip("\n") for line in text if line.strip()]


def _load_coco_rows(count: int, allow_download: bool) -> Iterator[tuple[np.ndarray, str]]:
    """Yield up to ``count`` (RGB uint8 image, caption) pairs from COCO.

    Fetches only the validation parquet shards (~850 MB) instead of the
    full 20 GB dataset, and stops decoding as soon as ``count`` rows have
    been produced.
    """
    from huggingface_hub import snapshot_download

    try:
        # Pinned to a commit SHA; bandit cannot resolve the constant.
        local_dir = snapshot_download(  # nosec B615
            repo_id=_COCO_REPO,
            repo_type="dataset",
            revision=_COCO_REVISION,
            allow_patterns="data/validation-*.parquet",
            local_files_only=not allow_download,
        )
    except OSError as e:
        # hf_hub offline/missing-file errors all derive from OSError.
        raise RealDataUnavailableError(_download_hint("COCO captions (validation split)")) from e

    shard_paths = sorted(Path(local_dir).glob("data/validation-*.parquet"))
    if not shard_paths:
        raise RealDataUnavailableError(_download_hint("COCO captions (validation split)"))

    import pyarrow.parquet as pq

    yielded = 0
    for shard_path in shard_paths:
        shard = pq.ParquetFile(shard_path)
        for batch in shard.iter_batches(batch_size=64, columns=["image", "caption"]):
            for row in batch.to_pylist():
                if yielded >= count:
                    return
                with Image.open(io.BytesIO(row["image"]["bytes"])) as decoded:
                    image = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
                yield image, str(row["caption"])
                yielded += 1


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class RealDataProvider:
    """Deterministic real data for benchmark scenarios, as raw numpy.

    Args:
        seed: Random seed controlling row selection order (default 42).
        data_dir: TFDS data directory override (default: ``TFDS_DATA_DIR``
            or ``~/tensorflow_datasets``).
        allow_download: Permit dataset downloads. Defaults to the
            ``DATARAX_BENCH_DOWNLOAD`` environment variable.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        data_dir: str | Path | None = None,
        allow_download: bool | None = None,
    ) -> None:
        """Initialize the provider with seed, data dir, and download policy."""
        self.rng = np.random.default_rng(seed)
        self.data_dir = Path(data_dir) if data_dir is not None else None
        if allow_download is None:
            allow_download = os.environ.get(DOWNLOAD_ENV_VAR) == "1"
        self.allow_download = allow_download

    def cifar10_images(self, n: int, h: int = 32, w: int = 32) -> np.ndarray:
        """Return n cifar10 train images, resized when (h, w) != (32, 32).

        Args:
            n: Number of images (tiles cyclically beyond the 50k train split).
            h: Output height in pixels.
            w: Output width in pixels.

        Returns:
            Array of shape ``(n, h, w, 3)`` with dtype uint8.
        """
        _require_positive(n)
        images = _load_cifar10_train(self.data_dir, self.allow_download)
        selected = images[_select_indices(len(images), n, self.rng)]
        return _resize_images(selected, h, w)

    def wikitext_tokens(self, n: int, seq_len: int, vocab_size: int = 32000) -> np.ndarray:
        """Return n fixed-length sequences packed from wikitext-103 text.

        Words are hash-tokenized (see :func:`hash_tokenize`) and packed
        contiguously into ``seq_len``-sized rows, preserving the corpus'
        natural word-frequency distribution. Rows are shuffled by seed.

        Args:
            n: Number of sequences (tiles the token stream if the corpus
                prefix is shorter than ``n * seq_len``).
            seq_len: Tokens per sequence.
            vocab_size: Ids are in ``[0, vocab_size)``.

        Returns:
            Array of shape ``(n, seq_len)`` with dtype int32.
        """
        _require_positive(n)
        target = n * seq_len
        chunks: list[np.ndarray] = []
        total = 0
        for line in _load_wikitext_lines(self.allow_download):
            words = line.split()
            if not words:
                continue
            chunks.append(hash_tokenize(words, vocab_size))
            total += len(words)
            if total >= target:
                break
        if not chunks:
            raise RealDataUnavailableError("wikitext-103 yielded no text lines")
        stream = np.resize(np.concatenate(chunks), target)
        sequences = stream.reshape(n, seq_len)
        return sequences[self.rng.permutation(n)]

    def criteo_features(self, n: int, hash_buckets: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
        """Return n Criteo DAC rows as DLRM-style dense + sparse arrays.

        Missing dense fields parse to 0.0; categorical fields are hashed
        into ``[0, hash_buckets)``.

        Args:
            n: Number of rows (tiles cyclically beyond the ~100k sample).
            hash_buckets: Bucket count for categorical hashing.

        Returns:
            Tuple of dense ``(n, 13)`` float32 and sparse ``(n, 26)`` int64.
        """
        _require_positive(n)
        dense_rows: list[list[float]] = []
        sparse_rows: list[np.ndarray] = []
        for line in _load_criteo_lines(self.allow_download):
            fields = line.split("\t")
            if len(fields) != _DAC_NUM_FIELDS:
                raise ValueError(
                    f"Malformed DAC line: expected {_DAC_NUM_FIELDS} "
                    f"tab-separated fields, got {len(fields)}"
                )
            dense_rows.append(
                [float(field) if field else 0.0 for field in fields[1 : 1 + _DAC_NUM_DENSE]]
            )
            sparse_rows.append(hash_tokenize(fields[1 + _DAC_NUM_DENSE :], hash_buckets))
        if not dense_rows:
            raise RealDataUnavailableError("Criteo DAC sample yielded no rows")
        dense = np.asarray(dense_rows, dtype=np.float32)
        sparse = np.stack(sparse_rows).astype(np.int64)
        indices = _select_indices(len(dense), n, self.rng)
        return dense[indices], sparse[indices]

    def coco_image_text(
        self,
        n: int,
        h: int = 64,
        w: int = 64,
        text_len: int = 77,
        vocab_size: int = 32000,
        image_dtype: str = "float32",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return n CLIP-style (image, caption-token) pairs from COCO.

        Args:
            n: Number of pairs (tiles cyclically beyond the ~25k split).
            h: Output image height.
            w: Output image width.
            text_len: Tokens per caption (truncated / zero-padded).
            vocab_size: Ids are in ``[0, vocab_size)``.
            image_dtype: ``"float32"`` for pixels scaled to [0, 1],
                ``"uint8"`` for raw pixel values.

        Returns:
            Tuple of images ``(n, h, w, 3)`` and tokens ``(n, text_len)`` int32.
        """
        _require_positive(n)
        if image_dtype not in ("float32", "uint8"):
            raise ValueError(f"image_dtype must be 'float32' or 'uint8', got {image_dtype!r}")
        images_list: list[np.ndarray] = []
        token_rows: list[np.ndarray] = []
        for image, caption in _load_coco_rows(n, self.allow_download):
            images_list.append(_resize_image(image, h, w))
            token_rows.append(_caption_tokens(caption, text_len, vocab_size))
        if not images_list:
            raise RealDataUnavailableError("COCO captions yielded no rows")
        indices = _select_indices(len(images_list), n, self.rng)
        images = np.stack(images_list)[indices]
        tokens = np.stack(token_rows)[indices]
        if image_dtype == "float32":
            return images.astype(np.float32) / 255.0, tokens
        return images, tokens
