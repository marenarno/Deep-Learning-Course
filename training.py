import os
from ultralytics import YOLO
import torch
import yaml

# Configuration
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
DATA_YAML_PATH = os.path.join(PROJECT_DIR, 'yolo_config.yaml')

BATCH_SIZE = 16
IMG_SIZE = 1024
PROJECT_NAME = 'runs/soccer_training'
RUN_NAME = 'exp'
DEVICE = '0' if torch.cuda.is_available() else 'cpu'

# Cleanup of GPU
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Load training config
with open("train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Starting YOLO training...")

# Stage 1
model = YOLO("yolov8s.pt")
model.train(**config["stage_1"])

# Stage 2
stage2_weights = os.path.join(config["stage_1"]["project"], config["stage_1"]["name"], "weights", "best.pt")
model = YOLO(stage2_weights)
model.train(**config["stage_2"])

# Stage 3
stage3_weights = os.path.join(config["stage_2"]["project"], config["stage_2"]["name"], "weights", "best.pt")
model = YOLO(stage3_weights)
model.train(**config["stage_3"])

print("Training complete.")
