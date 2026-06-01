import json
import os
from PIL import Image

INPUT_JSON = "sample-faces.json"
IMG_DIR = "images"
LABEL_DIR = "labels"

os.makedirs(LABEL_DIR, exist_ok=True)

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

for item in data:
    img_id = item.get("id")
    landmarks = item.get("frontLandmarks")

    if not img_id or not landmarks:
        continue

    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    if not os.path.exists(img_path):
        continue  # skip image + label

    try:
        img = Image.open(img_path)
        w, h = img.size
    except Exception:
        continue

    xs, ys = [], []
    kpts = []

    for p in landmarks.values():
        x = p["x"]
        y = p["y"]

        xs.append(x)
        ys.append(y)

        # normalize keypoints
        kpts.extend([
            x / w,
            y / h,
            2  # visibility
        ])

    # bounding box derived from keypoints (YOLO requires it)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    xc = ((x_min + x_max) / 2) / w
    yc = ((y_min + y_max) / 2) / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h

    label_line = "0 {:.6f} {:.6f} {:.6f} {:.6f} ".format(xc, yc, bw, bh)
    label_line += " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in kpts)

    label_path = os.path.join(LABEL_DIR, f"{img_id}.txt")
    with open(label_path, "w") as f:
        f.write(label_line + "\n")

print("YOLO keypoint labels created")
