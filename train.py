import os
import random
import shutil
import yaml
from ultralytics import YOLO

# ===================== CONFIG =====================
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_RATIO = 0.8
EPOCHS = 200
IMG_SIZE = 640
BATCH = 4  # CPU safe
WORKERS = 2

MODEL_NAME = "face_kpts_cpu"
PROJECT_DIR = "runs_pose"

# ===================== TRAIN / VAL SPLIT =====================
train_img_dir = os.path.join(IMAGES_DIR, "train")
val_img_dir = os.path.join(IMAGES_DIR, "val")
train_lbl_dir = os.path.join(LABELS_DIR, "train")
val_lbl_dir = os.path.join(LABELS_DIR, "val")

for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

def move_files(files, img_dst, lbl_dst):
    for img in files:
        src_img = os.path.join(IMAGES_DIR, img)
        src_lbl = os.path.join(LABELS_DIR, img.replace(".jpg", ".txt"))

        if not os.path.exists(src_lbl):
            continue

        shutil.move(src_img, os.path.join(img_dst, img))
        shutil.move(src_lbl, os.path.join(lbl_dst, img.replace(".jpg", ".txt")))

move_files(train_images, train_img_dir, train_lbl_dir)
move_files(val_images, val_img_dir, val_lbl_dir)

print("Train/Val split completed")

# ===================== AUTO-GENERATE data.yaml =====================
# Infer number of keypoints from one label file
sample_label = None
for f in os.listdir(train_lbl_dir):
    if f.endswith(".txt"):
        sample_label = os.path.join(train_lbl_dir, f)
        break

if sample_label is None:
    raise RuntimeError("No label files found")

with open(sample_label, "r") as f:
    parts = f.readline().strip().split()

num_kpts = (len(parts) - 5) // 3  # YOLO pose format

data_yaml = {
    "path": DATASET_DIR,
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["face"],
    "kpt_shape": [num_kpts, 3]
}

yaml_path = os.path.join(DATASET_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"data.yaml generated with {num_kpts} keypoints")

# ===================== CPU-ONLY YOLO TRAINING =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # hard-disable CUDA

model = YOLO("yolov8s-pose.pt")

model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device="cpu",
    workers=WORKERS,
    optimizer="AdamW",
    lr0=1e-3,
    cos_lr=True,
    patience=50,
    save=True,
    save_period=10,
    project=PROJECT_DIR,
    name=MODEL_NAME,
    exist_ok=True
)

print("Training completed")
