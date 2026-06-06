import json
import os
import requests

INPUT_JSON = "sample-faces.json"
IMG_DIR = "images"

os.makedirs(IMG_DIR, exist_ok=True)

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

downloaded_ids = set()

for item in data:
    url = item.get("frontPhotoUrl")
    img_id = item.get("id")
    if not url or not img_id:
        continue

    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(r.content)
        downloaded_ids.add(img_id)
    except Exception:
        pass

print(f"Downloaded {len(downloaded_ids)} images")
