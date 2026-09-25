import os
import json
import shutil
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET = "wuji3/face-recognition"
NUM_CLASSES = 20
IMAGES_PER_CLASS = 100
TRAIN_RATIO = 0.8

CACHE_DIR = "data/parquet_cache"
OUTPUT_DIR = "data/dataset_cnn"
CLASS_NAMES_FILE = "data/class_names.json"

os.makedirs("data", exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_and_standardize_face(image_bytes, desired_size=(224, 224)):
    """Cắt mặt tự động & chuyển đổi sang RGB chuẩn hóa."""
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            pad_w, pad_h = int(w * 0.15), int(h * 0.15)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(img_bgr.shape[1], x + w + pad_w), min(img_bgr.shape[0], y + h + pad_h)
            img_bgr = img_bgr[y1:y2, x1:x2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, desired_size, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(resized)

def read_shard(shard_id):
    path = hf_hub_download(
        repo_id=DATASET, repo_type="dataset",
        filename=f"data/train-{shard_id:05d}-of-00080.parquet", cache_dir=CACHE_DIR
    )
    return pd.read_parquet(path, columns=["image", "label", "class_name", "file_name"])

print("=" * 60 + "\nPHASE 1: LỌC CLASS ĐỦ MẪU\n" + "=" * 60)
class_counts = {}
for shard_id in range(80):
    df = read_shard(shard_id)
    for label, count in df["label"].value_counts().items():
        label = int(label)
        class_counts[label] = class_counts.get(label, 0) + int(count)
    eligible = [l for l, c in class_counts.items() if c >= IMAGES_PER_CLASS]
    if len(eligible) >= NUM_CLASSES:
        break

selected_labels = sorted(eligible[:NUM_CLASSES])
print(f"Đã chọn {len(selected_labels)} classes: {selected_labels}")

print("\n" + "=" * 60 + "\nPHASE 2: GOM VÀ TIỀN XỬ LÝ ẢNH\n" + "=" * 60)
samples, selected_counts, class_names = [], {l: 0 for l in selected_labels}, {}

for shard_id in range(80):
    if all(selected_counts[l] >= IMAGES_PER_CLASS for l in selected_labels):
        break
    df = read_shard(shard_id)
    filtered = df[df["label"].isin(selected_labels)]
    for _, row in filtered.iterrows():
        label = int(row["label"])
        if selected_counts[label] >= IMAGES_PER_CLASS:
            continue
        class_names[label] = str(row["class_name"])
        samples.append({
            "image": row["image"], "label": label,
            "class_name": str(row["class_name"]), "file_name": str(row["file_name"])
        })
        selected_counts[label] += 1

df_all = pd.DataFrame(samples)
labels = sorted(df_all["label"].unique())
label_to_id = {int(l): int(i) for i, l in enumerate(labels)}
df_all["class_id"] = df_all["label"].map(label_to_id)

with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
    json.dump({str(label_to_id[int(l)]): {"original_label": int(l), "class_name": class_names[int(l)]} for l in labels}, f, indent=4)

train_parts, val_parts = [], []
for l in labels:
    c_df = df_all[df_all["label"] == l]
    t_df, v_df = train_test_split(c_df, train_size=TRAIN_RATIO, random_state=42)
    train_parts.append(t_df)
    val_parts.append(v_df)

train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)

def export_to_folder(df_split, split_name):
    print(f"\nĐang xuất tập dữ liệu {split_name.upper()}...")
    for idx, row in tqdm(df_split.iterrows(), total=len(df_split)):
        c_id = int(row["class_id"])
        out_dir = os.path.join(OUTPUT_DIR, split_name, f"{c_id:02d}")
        os.makedirs(out_dir, exist_ok=True)
        img_out_path = os.path.join(out_dir, f"{split_name}_{idx:05d}.jpg")
        
        try:
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            processed_img = crop_and_standardize_face(img_bytes)
            processed_img.save(img_out_path, "JPEG", quality=95)
        except Exception as e:
            print(f"Lỗi ảnh {row['file_name']}: {e}")

export_to_folder(train_df, "train")
export_to_folder(val_df, "val")
print("\nHoàn tất chuẩn bị dữ liệu chuẩn PyTorch tại:", OUTPUT_DIR)