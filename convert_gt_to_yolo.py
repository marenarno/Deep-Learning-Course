import os
import yaml
import torch
import shutil

from config import (
    RAW_DATA_DIR,
    TRAIN_SRC_DIR_1,
    TRAIN_SRC_DIR_2,
    VAL_SRC_DIR,
    TEST_SRC_DIR
)

'''
This script converts annotations from the MOTChallenge format (gt.txt + labels.txt)
into YOLO format, where each frame gets a separate .txt file containing normalized
bounding boxes.

Input:
- gt.txt: Contains object annotations per frame (frame number, bounding box, class ID, etc.)
- labels.txt: Maps class IDs to class names (e.g., player, ball)

Output:
- For each frame, a corresponding .txt file is created with YOLO format:
  <class_id> <x_center> <y_center> <width> <height>
- All output label files are saved in a "labels/" folder inside each dataset path.

The script also prints a count of detected players and balls per frame for debugging purposes.

Usage:
Run the script directly to convert annotations for all test sequences defined in TEST_SRC_DIR.
"""
'''


TRAIN_SRC_DIRS = [TRAIN_SRC_DIR_1, TRAIN_SRC_DIR_2]

def convert_gt_to_yolo(gt_txt_path, labels_txt_path, output_dir, img_width=1920, img_height=1080):
    """
    Converts gt.txt + labels.txt to YOLO format labels.
    Creates one .txt file per frame with normalized bounding boxes.
    Also prints the count and coordinates of players and balls in each frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Load class names from labels.txt
    with open(labels_txt_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    class_dict = {i + 1: name for i, name in enumerate(classes)}  # class_id in gt.txt is 1-indexed
    name_to_index = {name: i for i, name in enumerate(classes)}  # for YOLO class IDs

    label_data = {}
    frame_objects = {}

    # Parse gt.txt
    with open(gt_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            class_id = int(parts[7])
            class_name = class_dict.get(class_id)
            if class_name not in name_to_index:
                continue

            yolo_id = name_to_index[class_name]

            # Convert to YOLO format
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w /= img_width
            h /= img_height

            line_out = f"{yolo_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
            frame_file = f"{frame:06}.txt"
            if frame_file not in label_data:
                label_data[frame_file] = []
                frame_objects[frame_file] = {'player': 0, 'ball': 0}
            label_data[frame_file].append(line_out)

            # Count for debug output
            if class_name == "player":
                frame_objects[frame_file]['player'] += 1
            elif class_name == "ball":
                frame_objects[frame_file]['ball'] += 1

    # Write YOLO label files
    for filename, lines in label_data.items():
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.writelines(lines)

    print(f"YOLO label files created at: {output_dir}\n")
    for frame_file, counts in sorted(frame_objects.items()):
        print(f"{frame_file}: {counts['player']} players, {counts['ball']} balls")


if __name__ == "__main__":
    for path in TEST_SRC_DIR:
        convert_gt_to_yolo(
            gt_txt_path=os.path.join(path, "gt", "gt.txt"),
            labels_txt_path=os.path.join(path, "gt", "labels.txt"),
            output_dir=os.path.join(path, "labels")
        )
