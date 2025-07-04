import os
import shutil
import random
import xml.etree.ElementTree as ET
from PIL import Image

# CONFIG
SRC_IMG_DIR = "data/raw_data/images"
SRC_XML_DIR = "data/raw_data/annotations"
DST_BASE = "data/pothole_data"
TRAIN_RATIO = 0.8
CLASS_MAPPING = {'pothole': 0}  # Modify if you have multiple classes

def clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

clean_folder(DST_BASE)

# Create target folders
for subdir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(DST_BASE, subdir), exist_ok=True)

# List all image files
image_files = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

# Split train/val
split_idx = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def voc_to_yolo(xml_path, img_size):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w, h = img_size
    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASS_MAPPING:
            continue
        cls_id = CLASS_MAPPING[cls_name]

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_c = ((xmin + xmax) / 2) / w
        y_c = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    return lines

def process_files(file_list, split):
    for img_name in file_list:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(SRC_IMG_DIR, img_name)
        xml_path = os.path.join(SRC_XML_DIR, base_name + '.xml')

        # Get image size
        img = Image.open(img_path)
        img_size = img.size  # (width, height)

        # Convert annotation
        yolo_lines = voc_to_yolo(xml_path, img_size)

        # Save image
        dst_img_path = os.path.join(DST_BASE, f'images/{split}/{img_name}')
        shutil.copy(img_path, dst_img_path)

        # Save label
        dst_lbl_path = os.path.join(DST_BASE, f'labels/{split}/{base_name}.txt')
        with open(dst_lbl_path, 'w') as f:
            for line in yolo_lines:
                f.write(line + '\n')

# Process train and val sets
process_files(train_files, 'train')
process_files(val_files, 'val')

# Save data.yaml
yaml_content = f"""
path: {DST_BASE}
train: images/train
val: images/val

names:
  0: pothole
"""

with open(os.path.join(DST_BASE, 'data.yaml'), 'w') as f:
    f.write(yaml_content.strip())

print("Conversion and splitting complete!")

def check_consistency(image_dir, label_dir):
    img_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))}
    lbl_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    missing_labels = img_files - lbl_files
    missing_images = lbl_files - img_files

    if missing_labels:
        print(f"Images with missing labels in {image_dir}:")
        for name in missing_labels:
            print(f"  {name}")

    if missing_images:
        print(f"Labels with missing images in {label_dir}:")
        for name in missing_images:
            print(f"  {name}")

    if not missing_labels and not missing_images:
        print(f"{image_dir} and {label_dir} are consistent.")
        print(f"Total images: {len(img_files)}, Total labels: {len(lbl_files)}")

# Run check for train and val
check_consistency(os.path.join(DST_BASE, 'images/train'), os.path.join(DST_BASE, 'labels/train'))
check_consistency(os.path.join(DST_BASE, 'images/val'), os.path.join(DST_BASE, 'labels/val'))