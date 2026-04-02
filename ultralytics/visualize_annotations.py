# visualize_annotations.py
import cv2
import os
import glob
import random


def plot_yolo_boxes(image_path, label_path, show=True, save_path=None, confidence_threshold=-1.0):
    img = cv2.imread(image_path)
    dh, dw, _ = img.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                continue

            if confidence_threshold >= 0.0:
                class_id, x, y, w, h, c  = map(float, line.split())
                if c < confidence_threshold:
                    continue
            else:
                class_id, x, y, w, h = map(float, line.split())

            if class_id == 0:
                color = (0, 255, 0)
            elif class_id == 1:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            # Convert YOLO to pixel coordinates
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            # Ensure coordinates are within image bounds
            l, r = max(0, l), min(dw - 1, r)
            t, b = max(0, t), min(dh - 1, b)

            cv2.rectangle(img, (l, t), (r, b), color, 2)

    if show:
        cv2.imshow('Annotations', img)
        cv2.waitKey(0)

    if save_path:
        cv2.imwrite(save_path, img)

    return img


#if __name__ == "__main__":
#    # Test random images
#    label_files = glob.glob("dataset_raw/labels/*.txt")  # TODO come from config
#    for label_file in random.sample(label_files, 5):  # Check 5 random samples
#        image_file = label_file.replace("labels", "images").replace(".txt", ".jpg")
#        if os.path.exists(image_file):
#            plot_yolo_boxes(image_file, label_file)

#    for specific_label in ["1", "2", "3", "4", "100", "1003", "1009"]:
#        label_file = f'dataset_raw/labels\\img{specific_label}.txt'
#        image_file = label_file.replace("labels", "images").replace(".txt", ".jpg")
#        if os.path.exists(image_file):
#            plot_yolo_boxes(image_file, label_file)