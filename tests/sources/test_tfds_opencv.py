"""The ``images`` extra: TFDS decodes images with OpenCV, and on one OpenCV thread in datarax.

TFDS's ``Image.decode_example_np`` decodes with OpenCV whenever ``cv2`` imports and with Pillow
otherwise, so installing the extra is what moves both TFDS sources to OpenCV, whose decode releases
the GIL. The two paths serve the same pixels for RGB and grayscale JPEGs and for PNGs, and differ
on CMYK JPEGs, pinned below so a codec change is seen. OpenCV's default pool, sized to the machine,
spent 2.2 to 2.8 times the CPU per decode of a single thread for no gain in wall time on 500x375
and 2000x1500 photos, so the TFDS sources decode on one OpenCV thread, as mmengine's worker set-up
does by default.
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow_datasets as tfds
from PIL import Image
from substrax.testing import run_python
from tensorflow_datasets.core.features import image_feature

from datarax.sources.decoded_copy import identity_differences
from datarax.sources.tfds_source import _per_batch_split
from datarax.sources.tfds_streaming_source import TFDSStreamingConfig, TFDSStreamingSource
from tests.jax_test_environment import forwarded_jax_environment
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture


pytestmark = pytest.mark.tfds

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_SECONDS = 300.0


def _photo(rows: int = 48, columns: int = 64) -> np.ndarray:
    """A smooth non-square RGB image, so a rotation would change its shape."""
    y, x = np.mgrid[0:rows, 0:columns]
    return np.stack([x * 4, y * 5, (x + y) * 2], axis=-1).astype(np.uint8)


def _encoded(image: Image.Image, fmt: str, **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


def _with_orientation(tag: int) -> bytes:
    exif = Image.Exif()
    exif[274] = tag
    return _encoded(Image.fromarray(_photo()), "JPEG", quality=90, exif=exif.tobytes())


_RGB = Image.fromarray(_photo())
_ENCODINGS: dict[str, tuple[bytes, int]] = {
    "jpeg baseline 4:4:4": (_encoded(_RGB, "JPEG", quality=90, subsampling=0), 3),
    "jpeg baseline 4:2:2": (_encoded(_RGB, "JPEG", quality=90, subsampling=1), 3),
    "jpeg baseline 4:2:0": (_encoded(_RGB, "JPEG", quality=90, subsampling=2), 3),
    "jpeg progressive": (_encoded(_RGB, "JPEG", quality=90, progressive=True), 3),
    "jpeg quality 75": (_encoded(_RGB, "JPEG", quality=75), 3),
    "jpeg grayscale, 3 channels": (_encoded(_RGB.convert("L"), "JPEG", quality=90), 3),
    "jpeg grayscale, 1 channel": (_encoded(_RGB.convert("L"), "JPEG", quality=90), 1),
    "mpo": (_encoded(_RGB, "MPO", quality=90), 3),
    "png rgb": (_encoded(_RGB, "PNG"), 3),
    "png rgba": (_encoded(_RGB.convert("RGBA"), "PNG"), 4),
    "png grayscale": (_encoded(_RGB.convert("L"), "PNG"), 1),
    **{f"jpeg exif orientation {tag}": (_with_orientation(tag), 3) for tag in range(1, 9)},
}


def _decoded(data: bytes, channels: int) -> tuple[np.ndarray, np.ndarray]:
    """``data`` through TFDS's OpenCV and Pillow paths, as ``decode_example_np`` calls them."""
    feature = tfds.features.Image(shape=(None, None, channels))
    return (
        feature.decode_example_np_with_opencv(data, channels),
        feature.decode_example_np_with_pil(data, channels),
    )


# The cv2 TFDS bound: the module imports no cv2 itself, so every lane collects it, the ones
# without the extra included; only the TFDS job, which installs the extra, runs it.
_OPENCV = image_feature.cv2


def test_tfds_decodes_images_with_opencv_once_the_extra_is_installed() -> None:
    assert _OPENCV is not None


@pytest.mark.parametrize("encoding", sorted(_ENCODINGS))
def test_opencv_and_pillow_serve_the_same_pixels(encoding: str) -> None:
    """Bit for bit, in shape and dtype too; EXIF orientation is applied by neither."""
    data, channels = _ENCODINGS[encoding]

    with_opencv, with_pillow = _decoded(data, channels)

    assert with_opencv.dtype == with_pillow.dtype == np.uint8
    assert with_opencv.shape == with_pillow.shape
    assert with_opencv.shape[:2] == (48, 64)
    np.testing.assert_array_equal(with_opencv, with_pillow)


def test_a_cmyk_jpeg_is_rgb_with_opencv_where_pillow_keeps_its_four_channels() -> None:
    """Pillow's path returns the file's CMYK channels, against the 3 the feature declares."""
    data = _encoded(_RGB.convert("CMYK"), "JPEG", quality=90)

    with_opencv, with_pillow = _decoded(data, 3)

    assert with_opencv.shape == (48, 64, 3)
    assert with_pillow.shape == (48, 64, 4)


class TestTheDecoderIdentity:
    """A decoded copy records the OpenCV that decoded it, so one written under Pillow differs."""

    def _identity(self, fixture: TFDSFixture) -> dict[str, Any]:
        split = _per_batch_split(FIXTURE, "train", data_dir=str(fixture.array_record))
        return split.source_identity()

    def test_names_the_installed_opencv(self, tfds_fixture: TFDSFixture) -> None:
        assert _OPENCV is not None
        assert self._identity(tfds_fixture)["decoder"]["opencv"] == _OPENCV.__version__

    def test_a_copy_decoded_under_pillow_differs_in_opencv_alone(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        current = self._identity(tfds_fixture)
        under_pillow = {**current, "decoder": {**current["decoder"], "opencv": None}}

        assert identity_differences(under_pillow, current) == [("decoder", "opencv")]


_THREADS_AFTER = """
import pickle, sys
import cv2
from substrax.testing import report

default = cv2.getNumThreads()
{decode}
report({{"default": default, "after": cv2.getNumThreads()}})
"""

_STREAM_DECODE = """
with open(sys.argv[1], "rb") as file:
    dataset = pickle.load(file)
next(iter(dataset))
"""

_EAGER_DECODE = """
from datarax.sources.tfds_source import TFDSEagerConfig, TFDSEagerSource
TFDSEagerSource(TFDSEagerConfig(name=sys.argv[1], split="train", data_dir=sys.argv[2]))
"""


def _threads_after(decode: str, *args: str) -> dict[str, int]:
    """OpenCV's pool in a fresh process before and after ``decode`` runs there."""
    child = run_python(
        _THREADS_AFTER.format(decode=decode),
        *args,
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )
    return child.check().report_as(dict[str, int])


@pytest.mark.skipif(os.cpu_count() == 1, reason="OpenCV's default pool is one thread here")
class TestOneOpenCVThread:
    """Each TFDS source decodes on one OpenCV thread in the process that decodes.

    The default is the machine's CPU count, so the default above one is the positive control.
    """

    def test_a_stream_s_dataset_read_in_a_worker_process(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        """A Grain worker receives the pickled dataset and iterates it, as here."""
        source = TFDSStreamingSource(
            TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(tfds_fixture.tfrecord))
        )
        pickled = tmp_path / "pass.pkl"
        # The dataset this test built, pickled as Grain pickles it for a worker; nothing else is
        # ever unpickled from this file.
        pickled.write_bytes(pickle.dumps(source.pass_dataset(0, None, 4)))

        threads = _threads_after(_STREAM_DECODE, str(pickled))

        assert threads["default"] > 1
        assert threads["after"] == 1

    def test_an_eager_source_s_build(self, tfds_fixture: TFDSFixture) -> None:
        threads = _threads_after(_EAGER_DECODE, FIXTURE, str(tfds_fixture.array_record))

        assert threads["default"] > 1
        assert threads["after"] == 1
