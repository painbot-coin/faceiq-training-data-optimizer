import json
import yaml
import os

INPUT_JSON = "sample-faces.json"
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# find first valid landmark set
num_kpts = None
for item in data:
    lm = item.get("frontLandmarks")
    if lm:
        num_kpts = len(lm)
        break

if num_kpts is None:
    raise RuntimeError("No landmarks found in JSON")

data_yaml = {
    "path": DATASET_DIR,
    "train": "images",
    "val": "images",
    "nc": 1,
    "names": ["face"],
    "kpt_shape": [num_kpts, 3]
}

with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"data.yaml created with {num_kpts} keypoints")
