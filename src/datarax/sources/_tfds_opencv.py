"""OpenCV, when TFDS decodes images with it, runs on one thread in each process that decodes.

TFDS's ``Image.decode_example_np`` decodes with OpenCV whenever ``cv2`` imports (datarax's
``images`` extra) and with Pillow otherwise, and converts each colour image with ``cv2.cvtColor``
(``tensorflow_datasets/core/features/image_feature.py``, 4.9.10). ``cvtColor`` runs on OpenCV's
own thread pool, which is sized to the machine unless ``OPENCV_FOR_THREADS_NUM`` sets it. Through
TFDS's path, that pool spent 2.2 to 2.8 times the CPU per decode of a single thread, for no gain
in wall time, on 500x375 and 2000x1500 photos. Every read worker that started it multiplied the
waste.

The loaders that run OpenCV in their worker processes switch the pool off: mmengine's
``set_multi_processing`` defaults ``opencv_num_threads=0`` "to avoid system being overloaded".
OpenCV runs ``parallel_for_`` in parallel only for a thread count below 0 or above 1
(``modules/core/src/parallel.cpp``), so one thread decodes sequentially. The setting is per
process, and it holds for the user's own OpenCV calls in that process too.

The module reads the ``cv2`` TFDS bound, the decoder TFDS chose, so it imports no ``cv2`` of its
own. The decoder identity (:mod:`datarax.sources._tfds_identity`) reads the same binding.
"""

from __future__ import annotations

from tensorflow_datasets.core.features import image_feature


def decode_on_one_opencv_thread() -> None:
    """Run OpenCV on one thread in this process, when TFDS decodes images with OpenCV.

    Each TFDS source calls it before it decodes a batch, wherever that runs (the training
    process, a read thread or a worker process). Once set, it costs one getter call.
    """
    cv2 = image_feature.cv2
    if cv2 is not None and cv2.getNumThreads() != 1:
        cv2.setNumThreads(1)
