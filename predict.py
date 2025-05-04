import os
from ultralytics import YOLO
from config import (
    PROJECT_DIR,
    TEST_IMAGE_PATH
)

'''
This script runs object detection predictions and evaluation on a test image set using a pre-trained YOLOv8 model.

Steps:
1. Loads the trained model from a specified .pt file.
2. Runs predictions on the test image directory and saves results (images + labels).
3. Evaluates the model using the test set defined in a YOLO-compatible data YAML file.
4. Prints key evaluation metrics (mAP, precision, recall).
'''

# Paths
MODEL_PATH = os.path.join(PROJECT_DIR, 'src', 'runs', 'soccer_training', 'exp_full_train', 'weights', 'best.pt')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'predictions', 'test')
DATA_YAML_PATH = os.path.join("..", "yolo_config.yaml")

# Load model
model = YOLO(MODEL_PATH)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
RUN_NAME = "test_predictions"

results = model.predict(
    source=TEST_IMAGE_PATH,
    save=True,
    save_txt=True,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    project=OUTPUT_DIR,
    name=RUN_NAME,
    exist_ok=True
)

print(f"Predictions saved to: {os.path.join(OUTPUT_DIR, 'test_predictions')}")

# Evaluate
metrics = model.val(data=DATA_YAML_PATH, split="test")
print("\n--- Evaluation Results on Test Set ---")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.precision:.3f}")
print(f"Recall: {metrics.box.recall:.3f}")
