"""Utility script to convert ImageNet64 dataset to ArrayRecord format.

This is a one-time data conversion utility, not a test file.
It requires TensorFlow which can hang on macOS ARM64 during import.
"""

import platform
from typing import NoReturn

# Guard against TensorFlow import hang on macOS ARM64 during pytest collection
# This is a known upstream issue: https://github.com/tensorflow/tensorflow/issues/52138
IS_MACOS = platform.system() == "Darwin"


def convert_to_array_record() -> str | NoReturn:
    """Convert the pickle-based ImageNet64 dataset to ArrayRecord.

    Returns:
        Path to the converted ArrayRecord file.

    Raises:
        RuntimeError: On macOS due to TensorFlow ARM64 import hang issue.
    """
    if IS_MACOS:
        raise RuntimeError(
            "This utility is not available on macOS due to TensorFlow ARM64 issues. "
            "Please run on Linux with TensorFlow installed."
        )

    # Import TensorFlow and other dependencies only when not on macOS
    import os
    import pickle

    import numpy as np
    import tensorflow as tf
    from array_record.python import array_record_module
    from pathlib import Path
    from tqdm import tqdm

    # Source path from user
    SOURCE_DATA_PATH = Path("/media/mahdi/ssd23/Data/SimplyHumanTakeHome/case_study_v2/to_share")
    # Output path (in our workspace)
    OUTPUT_DIR = Path("tests/data/imagenet64_arrayrecord")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading data from {SOURCE_DATA_PATH}")

    # Logic adapted from the user's data.py
    train_dir = SOURCE_DATA_PATH / "data/train_data"
    train_files = sorted(os.listdir(train_dir))

    output_path = OUTPUT_DIR / "imagenet64.arrayrecord"

    if output_path.exists():
        print(f"Output file {output_path} already exists. Skipping conversion.")
        return str(output_path)

    print(f"Writing to {output_path}")
    writer = array_record_module.ArrayRecordWriter(str(output_path), "group_size:1")

    total_records = 0

    # Process files one by one to avoid OOM during conversion (unlike the original script)
    for train_file in tqdm(train_files, desc="Converting files"):
        file_path = train_dir / train_file
        with open(file_path, "rb") as fo:
            data = pickle.load(fo)

            # Extract data and labels
            # Original shape: (N, 3072) ? or (N, 3, 64, 64) flattened?
            # data.py says: data["data"].reshape((N, 3, 64, 64)).transpose((0, 2, 3, 1))

            images = data["data"]
            labels = np.array(data["labels"])

            num_images = images.shape[0]

            for i in range(num_images):
                # We save the raw bytes or simple dictionary to ArrayRecord
                # To emulate "realistic" usage, we'll serialize a simple dictionary
                # containing the image bytes and the label.

                # Reshape just to verify, but save compact if possible.
                # Actually, saving as bytes is efficient.
                # But to properly test Grain's decoding, let's save a serialized example.
                # For simplicity and speed in this benchmark utility, we'll use
                # tf.train.Example which is standard for Grain/ArrayRecord usage.

                # Get the image data
                img_flat = images[i]  # This is a flat numpy array
                label = labels[i]

                # Create a feature
                feature = {
                    "image": tf.train.Feature(float_list=tf.train.FloatList(value=img_flat)),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())
                total_records += 1

    writer.close()
    print(f"Converted {total_records} records to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    convert_to_array_record()
