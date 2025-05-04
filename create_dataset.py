import os
import shutil
from glob import glob
from utils.dataset_utils import copy_files

'''
This script prepares training, validation, and test datasets for the project.
It organizes image and label files from raw directories into train/val/test folders.

Structure:
- All files from sequence 1 are used for training.
- Sequence 2 is split evenly into training and validation.
- Sequence 3 is used for testing.

Directory output:
dataset/
    └── train/
        ├── images/
        └── labels/
    └── val/
        ├── images/
        └── labels/
    └── test/
        ├── images/
        └── labels/
"""
'''

from config import (
    TRAIN_SRC_DIR_1,
    TRAIN_SRC_DIR_2,
    TEST_SRC_DIR,

    TRAIN_IMAGE_PATH,
    TRAIN_LABEL_PATH,
    VAL_IMAGE_PATH,
    VAL_LABEL_PATH,
    TEST_IMAGE_PATH,
    TEST_LABEL_PATH
)


def prepare_datasets():
    print("Preparing training, validation, and test datasets...")

    # Clean up existing directories
    for path in [TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, VAL_IMAGE_PATH, VAL_LABEL_PATH, TEST_IMAGE_PATH, TEST_LABEL_PATH]:
        if os.path.exists(path):
            shutil.rmtree(path)

    # Video file 1 for training
    img_dir_1 = os.path.join(TRAIN_SRC_DIR_1, 'img1')
    lbl_dir_1 = os.path.join(TRAIN_SRC_DIR_1, 'labels')
    label_files_1 = [os.path.basename(p) for p in glob(os.path.join(lbl_dir_1, '*.txt')) if 'labels.txt' not in p]
    copy_files(img_dir_1, lbl_dir_1, TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, label_files_1, prefix="1")

    # Half of video file 2 for training and half for validation
    img_dir_2 = os.path.join(TRAIN_SRC_DIR_2, 'img1')
    lbl_dir_2 = os.path.join(TRAIN_SRC_DIR_2, 'labels')
    label_files_2 = sorted([os.path.basename(p) for p in glob(os.path.join(lbl_dir_2, '*.txt')) if 'labels.txt' not in p])
    split_idx = len(label_files_2) // 2

    train_files_2 = label_files_2[:split_idx]
    val_files_2 = label_files_2[split_idx:]
    copy_files(img_dir_2, lbl_dir_2, TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, train_files_2, prefix="2")
    copy_files(img_dir_2, lbl_dir_2, VAL_IMAGE_PATH, VAL_LABEL_PATH, val_files_2, prefix="2")

    # Video file 2 for test
    img_dir_3 = os.path.join(TEST_SRC_DIR, 'img1')
    lbl_dir_3 = os.path.join(TEST_SRC_DIR, 'labels')
    label_files_3 = [os.path.basename(p) for p in glob(os.path.join(lbl_dir_3, '*.txt')) if 'labels.txt' not in p]
    copy_files(img_dir_3, lbl_dir_3, TEST_IMAGE_PATH, TEST_LABEL_PATH, label_files_3, prefix="3")

    print("Training, validation, and test datasets prepared.")
    print(f"Training images: {TRAIN_IMAGE_PATH}")
    print(f"Validation images: {VAL_IMAGE_PATH}")
    print(f"Test images: {TEST_IMAGE_PATH}")

if __name__ == "__main__":
    prepare_datasets()
