# Copy images from /dataset_raw/ to /dataset_fix/ and splits it in /dataset/ (WARNING: OVERRIDES BOTH FOLDERS)

import glob
import os
from PIL import Image
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolo_buildings")))
from visualize_annotations import plot_yolo_boxes


# Destination folders  # TODO also use these on the saving part (end of file) WIP
dst_base = "dataset_single_crop_1class"  #"dataset_fix"
dst_images = os.path.join(dst_base, "images")
dst_labels = os.path.join(dst_base, "labels")
dst_debug_view = os.path.join(dst_base, "images_with_labels")

add_center_crop = False
skip_extra_crops = True
force_max_crop_size = False  # This will keep original image size
add_original_image = True
crop_size = 640  # Only used if force_max_crop_size=false
rescale_size = 640


# Reset dataset folder
if os.path.exists(dst_base):
    shutil.rmtree(dst_base)

# Create destination folders
for folder in [dst_images, dst_labels, dst_debug_view]:
    os.makedirs(folder, exist_ok=True)


label_files = glob.glob("dataset_raw/labels/*.txt")
image_files = glob.glob("dataset_raw/images/*.jpg")

img_sizes = []

for label_file in label_files:
    stem = os.path.splitext(os.path.basename(label_file))[0] + "."
    matching_images = [img for img in image_files if stem in img]

    for image_file in matching_images:
        # C
        img = Image.open(image_file)

        if force_max_crop_size:
            crop_size = min(img.size)

        assert crop_size <= img.size[0] and crop_size <= img.size[1], \
            "ERROR: Original image too small for selected crop size"


        # if not img.size in img_sizes:
        #     img_sizes.append(img.size)
        #     print(img_sizes)

        left = int((img.size[0] - crop_size) / 2)
        top = int((img.size[1] - crop_size) / 2)
        right = left + crop_size
        bottom = top + crop_size

        crop_limits = []
        if add_center_crop:
            crop_limits.append((left,top,right,bottom))
        if skip_extra_crops:
            pass
        elif img.size == (1024, 1024):
            for left, top in zip([0, 0, img.size[0] - crop_size, img.size[0] - crop_size], [0, img.size[1] - crop_size, 0, img.size[1] - crop_size]):
                right = left + crop_size
                bottom = top + crop_size
                crop_limits.append((left, top, right, bottom))

        elif img.size == (768, 1024):
            for top in [0, img.size[1] - crop_size]:
                left = int((img.size[0] - crop_size) / 2)
                right = left + crop_size
                bottom = top + crop_size
                crop_limits.append((left, top, right, bottom))

        elif img.size == (1024, 768):
            for left in [0, img.size[0] - crop_size]:
                top = int((img.size[1] - crop_size) / 2)
                right = left + crop_size
                bottom = top + crop_size
                crop_limits.append((left, top, right, bottom))

        elif img.size == (768, 768):
            pass  # Already have center in the list

        else:
            print(f"WARNING: unexpected image size for {image_file}, extracting only center")

        for idx in range(len(crop_limits)):
            left, top, right, bottom = crop_limits[idx]


            left_perc = left / img.size[0]
            right_perc = right / img.size[0]
            top_perc = top / img.size[1]  # top = small y
            bottom_perc = bottom / img.size[1]  # bottom = large y

            labels = []
            str_labels = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # skip malformed lines
                    _, x_center, y_center, width, height = map(float, parts)

                    center_in_crop = left_perc <= x_center <= right_perc and top_perc <= y_center <= bottom_perc

                    full_bb_in_crop_x = left_perc <= x_center - width/2.0 and  right_perc >= x_center + width/2.0
                    full_bb_in_crop_y = top_perc <= y_center - height/2.0 and  bottom_perc >= y_center + height/2.0
                    full_bb_in_crop = full_bb_in_crop_x and full_bb_in_crop_y

                    any_bb_in_crop_x = (x_center - width / 2.0 < right_perc) and (x_center + width / 2.0 > left_perc)
                    any_bb_in_crop_y = (y_center - height / 2.0 < bottom_perc) and (y_center + height / 2.0 > top_perc)
                    any_bb_in_crop = any_bb_in_crop_x and any_bb_in_crop_y

                    my_class = 0 if full_bb_in_crop else 1 if center_in_crop else 2

                    # Only keep bbs that have center inside the cropped box  # TODO change? Full bb inside crop?
                    if not full_bb_in_crop :  #not any_bb_in_crop:
                        pass

                    else:

                        # Compute original edges
                        x_min = x_center - width / 2
                        x_max = x_center + width / 2
                        y_min = y_center - height / 2
                        y_max = y_center + height / 2

                        # Determine class
                        if x_min >= left_perc and x_max <= right_perc and y_min >= top_perc and y_max <= bottom_perc:
                            bb_class = 0  # fully inside
                        elif x_center >= left_perc and x_center <= right_perc and y_center >= top_perc and y_center <= bottom_perc:
                            bb_class = 1  # center inside, BB partially outside
                        else:
                            bb_class = 2  # center outside, BB partially inside
                        assert(bb_class == my_class)

                        # Clip edges to crop boundaries
                        x_min_clipped = max(x_min, left_perc)
                        x_max_clipped = min(x_max, right_perc)
                        y_min_clipped = max(y_min, top_perc)
                        y_max_clipped = min(y_max, bottom_perc)

                        # Compute new center
                        if bb_class == 2:
                            # center must be inside crop
                            x_center_new = min(max(x_center, left_perc), right_perc)
                            y_center_new = min(max(y_center, top_perc), bottom_perc)
                        else:
                            # keep original center
                            x_center_new = x_center
                            y_center_new = y_center

                        # Width/height adjusted to visible part
                        width_new = x_max_clipped - x_min_clipped
                        height_new = y_max_clipped - y_min_clipped

                        # if width_new * height_new < width * height * .33:  # Class 2 minimum area parameter (keeping only if area >33%)
                        #     continue

                        # Normalize relative to crop
                        x_center_new = (x_center_new - left_perc) / (right_perc - left_perc)
                        y_center_new = (y_center_new - top_perc) / (bottom_perc - top_perc)
                        width_new /= (right_perc - left_perc)
                        height_new /= (bottom_perc - top_perc)

                        labels.append({
                            "class": int(my_class),
                            "x_center": float(x_center_new),
                            "y_center": float(y_center_new),
                            "width": float(width_new),
                            "height": float(height_new)
                        })

                        str_labels.append(" ".join([
                            str(int(my_class)),
                            str(float(x_center_new)),
                            str(float(y_center_new)),
                            str(float(width_new)),
                            str(float(height_new))]))

            # Adjust image
            right  = left + crop_size
            bottom = top + crop_size

            crop_box = (left, top, right, bottom)
            cropped = img.crop(crop_box)

            file_out_name = image_file.split("\\")[-1].split(".")[0] + f"_{idx}."

            cropped_resized = cropped.resize((rescale_size, rescale_size), Image.LANCZOS)
            cropped_image_path = f"{dst_base}\\images\\" + file_out_name + image_file.split("\\")[-1].split(".")[-1]
            cropped_resized.save(cropped_image_path)

            cropped_label_path = f"{dst_base}\\labels\\" + file_out_name + label_file.split("\\")[-1].split(".")[-1]
            with open(cropped_label_path, "w") as f_out:
                f_out.write("\n".join(str_labels) + "\n")  # add final newline


            # VIS
            # plot_yolo_boxes(image_file, label_file)
            # plot_yolo_boxes(cropped_image_path, cropped_label_path)
            debug_vis_path = f"{dst_base}\\images_with_labels\\" + file_out_name + image_file.split("\\")[-1].split(".")[-1]
            plot_yolo_boxes(cropped_image_path, cropped_label_path, False, debug_vis_path)

        if add_original_image:
            # Image
            og_image_path = f"{dst_base}\\images\\" + image_file.split("\\")[-1]
            img.save(og_image_path)
            # Label
            og_labels = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # skip malformed lines
                    cls, x_center, y_center, width, height = map(float, parts)
                    if not (0.0 <= x_center <= 1.0) or not (0.0 <= y_center <= 1.0):
                        # print("Center outside of BB!")
                        # print(x_center, y_center, width, height)
                        x_center = min(max(x_center, 0.0), 1.0)
                        y_center = min(max(y_center, 0.0), 1.0)
                        cls = 1
                        continue
                    elif x_center + width > 1.0 or x_center - width < 0.0 or y_center + height > 1.0 or y_center - height < 0.0:
                        # print("Partial BB!")
                        csl = 1
                        continue
                    og_labels_row = [
                        str(int(cls)),
                        str(float(x_center)),
                        str(float(y_center)),
                        str(float(width)),
                        str(float(height))
                    ]

                    og_labels.append((" ".join(og_labels_row)))

            og_label_path = f"{dst_base}\\labels\\" + label_file.split("\\")[-1]
            with open(og_label_path, "w") as f_out:
                f_out.write("\n".join(og_labels) + "\n")  # add final newline
            # Debug
            debug_vis_path = f"{dst_base}\\images_with_labels\\" + image_file.split("\\")[-1]
            plot_yolo_boxes(image_file, label_file, False, debug_vis_path)