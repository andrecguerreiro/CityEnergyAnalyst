# smoke_test_model.py
import torch
from pathlib import Path
from ultralytics.nn.tasks import RoofModel
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
import cv2
import yaml, torch
from ultralytics.data import build_dataloader
from ultralytics.data.augment import v8_transforms
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics import YOLO

def plot_yolo_boxes(img, predictions, confidence_threshold=-1.0):
    dh, dw, _ = img.shape
    for i in range(predictions.shape[0]):  
        x1, y1, x2, y2, c, classid  = predictions[i,:]
        if c < confidence_threshold:
            continue
        color = (0, 255, 0)
        # Ensure coordinates are within image bounds
        l, r = max(0, int(x1)), min(dw - 1, int(x2))
        t, b = max(0, int(y1)), min(dh - 1, int(y2))

        cv2.rectangle(img, (l, t), (r, b), color, 2)
    return img

overrides = {
    "data": "datasets/roof_dataset.yaml",
    "model": "yolo11n-roof.yaml",   # or path to model config / weights
    "epochs": 1,
    "batch": 2,
    "task" : "roof"
}

args = get_cfg(DEFAULT_CFG, overrides)  # or a custom dict with required hyperparameters
args.rooftop = True
data = check_det_dataset("datasets/roof_dataset.yaml")
dataset = build_yolo_dataset(args , img_path= data['train'], batch = 2 , data = data, mode="train", rect=False, stride=32)

model = YOLO("best.pt")
model.to("cpu")
model.eval()

dataminus = dataset[0]
img = dataminus['img'].to(float)
predictions = model(img.data)
print("testing purpousus")