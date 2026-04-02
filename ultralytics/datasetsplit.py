import os
import shutil
from glob import glob
import re

def natural_sort_key(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    nums = re.findall(r'\d+', stem)
    return [int(n) for n in nums] if nums else [0]

src_images = "C:/Dev/BeNeutral/datasetgeneration/tiny_dataset/perfecthandmadewithdepth/images"
src_labels = "C:/Dev/BeNeutral/datasetgeneration/tiny_dataset/perfecthandmadewithdepth/labels"

dst_base = "datasets/roof_dataset"
dst_images_train = os.path.join(dst_base, "images/train")
dst_images_val   = os.path.join(dst_base, "images/val")
dst_labels_train = os.path.join(dst_base, "labels/train")
dst_labels_val   = os.path.join(dst_base, "labels/val")

if os.path.exists(dst_base):
    shutil.rmtree(dst_base)

for folder in [dst_images_train, dst_images_val, dst_labels_train, dst_labels_val]:
    os.makedirs(folder, exist_ok=True)

image_files = sorted(glob(os.path.join(src_images, "*.*")), key=natural_sort_key)
num_images = len(image_files)
split_idx = int(num_images * 0.8)  # 80% train

train_images = image_files[:split_idx]
val_images   = image_files[split_idx:]

def append_zero_lines_to_label(src_label_path, dst_label_path, num_lines=13):
    lines_to_write = []
    appendix = " ".join(["0"] * num_lines)  # "0 0 0 ...
    if os.path.exists(src_label_path):
        with open(src_label_path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                lines_to_write.append(f"{line} {appendix}\n")
        with open(dst_label_path, 'w') as out:
            out.writelines(lines_to_write)
    
def copy_files(image_list, dst_image_folder, dst_label_folder):
    for img_path in image_list:
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        # Copy image
        shutil.copy2(img_path, os.path.join(dst_image_folder, filename))
        # Copy corresponding label

        src_label = os.path.join(src_labels, stem + ".txt")
        dst_label = os.path.join(dst_label_folder, stem + ".txt")
        append_zero_lines_to_label(src_label, dst_label)

copy_files(train_images, dst_images_train, dst_labels_train)
copy_files(val_images, dst_images_val, dst_labels_val)

print(f"Copied {len(train_images)} images to train, {len(val_images)} images to val (total: {num_images})")