import os
import cv2
import datetime
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import Button
import numpy as np
import json
from sympy import false
from ultralytics import YOLO
import glob
import shutil
import msvcrt

# Helper: normalize/save label file (remove confidence column if present)
def write_label_file(source_label_path: Path, dest_label_path: Path, strip_confidence: bool = False):
    """
    Copies or rewrites a label file to dest_label_path.
    If strip_confidence=True, drop the last column (confidence) from each label line.
    """
    if not source_label_path.exists():
        # create empty label
        dest_label_path.write_text("")
        return

    lines = source_label_path.read_text().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if strip_confidence and len(parts) >= 6:
            # assume format: class x y w h conf  (or class + xywhn + conf)
            new_lines.append(" ".join(parts[:-1]))
        else:
            new_lines.append(" ".join(parts))
    dest_label_path.write_text("\n".join(new_lines))

# Unified saving routine
def save_to_bucket(bucket_name: str, image_path: Path, label_path: Path, prediction_txt: Path = None, use_prediction: bool = False):
    """
    Save image and label into bucket_name folder.
    If use_prediction=True and prediction_txt exists, the prediction file (with confidence column)
    will be used as the saved label (confidence stripped).
    Otherwise label_path (original) is used.
    """
    base = DATASET_BUCKETS[bucket_name]
    img_dst = base / "images" / image_path.name
    lbl_dst = base / "labels" / Path(label_path).name

    # copy image
    shutil.copyfile(image_path, img_dst)

    if use_prediction and prediction_txt is not None and prediction_txt.exists():
        # write prediction as label without confidence
        write_label_file(prediction_txt, lbl_dst, strip_confidence=True)
    else:
        # copy (or rewrite) the original label
        write_label_file(label_path, lbl_dst, strip_confidence=False)

# MINE ERASE sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolo_buildings")))
from visualize_annotations import plot_yolo_boxes

# ----------------------
# Placeholder functions
# ----------------------
def get_corresponding_label(image_path, where="labels"):
    label_path = Path(str(image_path).replace("images", where)).with_suffix(".txt")
    return label_path

def get_with_labels(image_path):
    path_str = str(image_path)
    img = plot_yolo_boxes(path_str, get_corresponding_label(path_str), False)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_with_labels(image_path):
    path_str = image_path
    img = plot_yolo_boxes(path_str, get_corresponding_label(path_str))
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_model_prediction(image_path, prediction_path):
    path_str = image_path
    pred_path_str = str(prediction_path)
    pred_img = plot_yolo_boxes(
        path_str,
        get_corresponding_label(
            pred_path_str,
            f"prediction_{model_name}"),
        False,
        None,
        confidence_threshold)
    return pred_img

def make_model_prediction(image_paths):
    for image_path in image_paths:
        image_path = str(image_path)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        results = model(image_path)

        print("Done")

def show_model_prediction(image_path):
    """Return an image with model prediction drawn (placeholder)."""
    img = cv2.imread(image_path)
    #
    # # Run inference on an image
    # results = model(img)
    results = model(image_path)

    # Loop through results
    for r in results:
        boxes = r.boxes.xyxy  # [x1, y1, x2, y2] in pixels
        confs = r.boxes.conf  # confidence scores
        clss = r.boxes.cls  # class IDs
        print(boxes, confs, clss)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------------------
# Setup
# -----------------------------------------------------------------
with open("dataset_config.json", "r") as f:
    config = json.load(f)

ROOT = Path(config['datasetpath'])  # contains train/ and val/
IMAGE_DIR = ROOT / "images"
temp_path_name = "testing"

DATASET_BUCKETS = {
    "perfect": ROOT / "perfect",
    "manual_review": ROOT / "manual_review",
    "very_wrong": ROOT / "very_wrong"
}


for name, base in DATASET_BUCKETS.items():
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "labels").mkdir(parents=True, exist_ok=True)

confidence_threshold = 0.5
model_name = "train17"
IMAGE_SIZE = 748
BATCH_SIZE = 16
# -----------------------------------------------------------------

# Load a pre-trained model
# model = YOLO("yolov11s.pt")   # built-in pretrained weights
# Or load your custom trained weights
# model = YOLO("runs/detect/train4/weights/best.pt")
# model = YOLO(f"D:\\src\\BeNeutral\\runs\\detect\\{model_name}\\weights\\best.pt")


model = YOLO(Path(config['modelpath'])  )
output_predictions_path = Path(config['datasetpath']+  f"prediction_{model_name}")
# delete if exists
if not output_predictions_path.exists():
    # shutil.rmtree(pred_dir)  # DELETE
    output_predictions_path.mkdir(parents=True, exist_ok=True)

# Lists to track decisions
keep_current = []
keep_new = []
manual_review = []
golden_current = []
golden_new = []

# Load golden list from file to skip them in future
GOLDEN_RECORD_FILE = Path(config['datasetpath'] + "golden_records.txt")
if os.path.exists(GOLDEN_RECORD_FILE):
    with open(GOLDEN_RECORD_FILE, "r") as f:
        # golden_paths = [[val.strip() for val in line.split("\t")][0].split("\\")[-1] for line in f]
        golden_paths = [[val.strip() for val in line.split("\t")][0].split("\\")[-1] for line in f]
else:
    golden_paths = set()

def on_perfect():
    # use model prediction as label if available, else fallback to original label
    #save_to_bucket("perfect", img_path, label_path, out_file_prediction, use_prediction=True)
    plt.close(fig)

def on_manual_review():
    #save_to_bucket("manual_review", img_path, label_path, out_file_prediction, use_prediction=True)
    plt.close(fig)

def on_very_wrong():
    #save_to_bucket("very_wrong", img_path, label_path, out_file_prediction, use_prediction=True)
    plt.close(fig)

# ----------------------
# Iteration
# ----------------------
last_accepted_img = None

# all_images = list(ROOT.glob("**/*.jpg")) + list(ROOT.glob("**/*.png"))
all_images = list(IMAGE_DIR.glob("**/*.jpg")) + list(IMAGE_DIR.glob("**/*.png"))
all_images_str = [str(p) for p in all_images]
# Model predicitons
# results = model(all_images_str, imgsz=640, batch=4)  # batch=8 (adjust for your GPU)
results = model.predict(
    # source=all_images_str,  # list of paths is fine
    source=IMAGE_DIR,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,                # adjust for GPU
    stream=False            # default, processes in chunks
)
# results is a list of Results objects, one per image
import torch as t
for img_path, r in zip(all_images, results):
    img_path_str = str(img_path)
    out_file_prediction = output_predictions_path / (Path(img_path).stem + ".txt")

    new_txt = t.cat(
        (r.boxes.cls.reshape(-1, 1),
            r.boxes.xywhn,
            r.boxes.conf.reshape(-1, 1)),
        dim=1
    )
    new_txt = new_txt[new_txt[:, -1] > confidence_threshold]

    #lines = [
    #    " ".join(map(str, row.tolist()))
    #    for row in new_txt.cpu()
    #]

    lines = [
        f"{int(row[0])} " + " ".join(f"{x:.10f}" for x in row[1:])
        for row in new_txt.cpu().tolist()
    ]

    out_file_prediction.write_text("\n".join(lines))

    #if str(img_path) in golden_paths:
    #    continue  # Skip golden dataset images

    folder = img_path.parent.name
    label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")

    # Build display grid
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    # a) current image + labels
    axes[0].imshow(get_with_labels(str(img_path)))
    axes[0].set_title("a) Current image + labels")
    axes[0].axis("off")

    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+50+50")  # "+X+Y" pixels from top-left of screen

    # b) model output
    axes[1].imshow(get_model_prediction(img_path_str, out_file_prediction))
    axes[1].set_title("b) Model prediction")
    axes[1].axis("off")

    plt.subplots_adjust(bottom=0.2)  # leave space at bottom for buttons

    # Button positions [left, bottom, width, height] in figure coordinates
    button_width = 0.15
    button_height = 0.05
    spacing = 0.02
    start_x = 0.1  # start left

    #button_labels = ["Perfect (use prediction)", "Manual Review", "Very Wrong"]
    #callbacks = [on_perfect, on_manual_review, on_very_wrong]

    def close_event():
        on_perfect()


    #buttons = []
    #for i, (label, cb) in enumerate(zip(button_labels, callbacks)):
    #    ax_btn = fig.add_axes([start_x + i * (button_width + spacing), 0.05, button_width, button_height])
    #    btn = Button(ax_btn, label)
    #    btn.on_clicked(cb)
    #    buttons.append(btn)  # keep a reference!
    #timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
    #timer.add_callback(close_event)
    #timer.start()
    plt.show(block = True)
    
    # plt.tight_layout()
    # plt.show(block=True)
    # key = input("Choose action [1: keep current, 3: keep new, 5: manual review, 7: keep current + golden, 9: keep new + golden]: ")
    #
    # # Decision logic
    # if key == "1":
    #     keep_current.append((img_path, label_path))
    #     # last_accepted_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    #
    # elif key == "3":
    #     keep_new.append((img_path, out_file_prediction))
    #     # last_accepted_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    #
    # elif key == "5":
    #     keep_current.append((img_path, label_path))
    #     manual_review.append((img_path, label_path))
    #
    # elif key == "7":
    #     golden_current.append((img_path, label_path))
    #     # golden_paths.add(str(img_path))
    #     # last_accepted_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    #
    # elif key == "9":
    #     golden_new.append((img_path, out_file_prediction))
    #     # golden_paths.add(str(img_path))
    #     # last_accepted_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

# ----------------------
# Save results
# ----------------------
#def save_list(list_of_tuples, subname):
#    out_file = ROOT / f"{subname}.txt"
#    with open(out_file, "w") as f:
#        for img, lbl in list_of_tuples:
#            f.write(f"{img},{lbl}\n")
#
# save_list(keep_current, "keep_current")
# save_list(keep_new, "keep_new")
#save_list(manual_review, "manual_review")
# save_list(golden_current, "golden_current")
# save_list(golden_new, "golden_new")

#golden_set = golden_current + golden_new
# Update golden records file
#with open(GOLDEN_RECORD_FILE, "w") as f:
#    for file in golden_paths:
#        f.write(str(file) + "\n")

#    for gjpg, gtxt in golden_set:
#        f.write(gjpg.name + "\n")
#        # f.write(str(gjpg) + "\t" + str(gtxt) + "\n")


# Make new /labels/
#dst_labels = ROOT / "labels"
# dst_labels.mkdir(parents=True, exist_ok=True)
# for move_list in [keep_current, keep_new]:
#     for _, label in move_list:
# for _, label in keep_new:
#         file_dst = dst_labels / label.name
#         shutil.copy(label, file_dst)

#for _, label in keep_new:
#    file_dst = dst_labels / label.name

    # Read lines
#    with open(label, "r") as f:
#        lines = f.readlines()

    # Remove last element of each line
#    new_lines = []
#    for line in lines:
#        parts = line.strip().split()
#        if len(parts) > 1:         # skip empty lines or single-element lines
#            new_lines.append(" ".join(parts[:-1]))
#        else:
#            new_lines.append("")    # or skip line entirely

#    # Write modified lines
#    file_dst.parent.mkdir(parents=True, exist_ok=True)
#    with open(file_dst, "w") as f:
#        f.write("\n".join(new_lines))


print(f"Finished. Results updated in {ROOT}")