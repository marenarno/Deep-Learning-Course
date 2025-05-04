
import os
from utils.image_utils import draw_boxes_on_frame
from config import (
    PROJECT_DIR,
    DATASET_DIR,
    RAW_DATA_DIR, 

    TRAIN_SRC_DIR_1,
    TRAIN_SRC_DIR_2
)
'''
This script loads a single image frame and its corresponding YOLO label file,
draws bounding boxes on the image, and saves the result as an annotated image.

Input:
- Image path: expects a .jpg frame with a 6-digit frame number (e.g., 000001.jpg)
- Label path: corresponding YOLO-format .txt file with class ID and bounding box info

Output:
- An image file with visualized bounding boxes saved as "frame_XXXXXX_labeled.jpg"
'''

TRAIN_SRC_DIRS = [TRAIN_SRC_DIR_1, TRAIN_SRC_DIR_2]

frame_number = 1
frame_str = f"{frame_number:06}"
img_path = os.path.join(TRAIN_SRC_DIRS[0], "img1", f"{frame_str}.jpg")
lbl_path = os.path.join(TRAIN_SRC_DIRS[0], "labels", f"{frame_str}.txt")
output_path = f"frame_{frame_str}_labeled.jpg"

draw_boxes_on_frame(img_path, lbl_path, ["ball", "player"], output_path)