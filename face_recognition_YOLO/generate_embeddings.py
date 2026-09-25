import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/dataset_cnn/train"
CLASS_NAMES_FILE = "data/class_names.json"
OUTPUT_FILE = "data/class_embeddings.pt"

# MTCNN(post_process=True) đã tự chuẩn hóa tensor cho FaceNet
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True, device=DEVICE)
cnn_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

num_classes = len(class_mapping)
class_centroids = [None] * num_classes

print(f"⚡ Đang trích xuất Vector Centroid cho {num_classes} classes...")

for str_idx, info in tqdm(class_mapping.items()):
    idx = int(str_idx)
    folder_name = f"{idx:02d}"
    class_path = os.path.join(DATA_DIR, folder_name)

    if not os.path.exists(class_path):
        class_path = os.path.join(DATA_DIR, str(info.get("original_label", idx)))

    embeddings_list = []
    if os.path.exists(class_path):
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    img = Image.open(os.path.join(class_path, img_name)).convert("RGB")
                    face_tensor = mtcnn(img)
                    
                    if face_tensor is not None:
                        with torch.no_grad():
                            emb = cnn_embedder(face_tensor.unsqueeze(0).to(DEVICE))
                            emb = F.normalize(emb, p=2, dim=1)
                            embeddings_list.append(emb)
                except Exception:
                    continue

    if embeddings_list:
        centroid = torch.mean(torch.cat(embeddings_list, dim=0), dim=0, keepdim=True)
        class_centroids[idx] = F.normalize(centroid, p=2, dim=1)
    else:
        class_centroids[idx] = F.normalize(torch.randn(1, 512), p=2, dim=1).to(DEVICE)

torch.save(torch.cat(class_centroids, dim=0), OUTPUT_FILE)
print(f"✅ Đã tạo thành công file Vector Centroids: {OUTPUT_FILE}")