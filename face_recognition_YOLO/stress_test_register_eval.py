import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
import pandas as pd

# ============================================================
# 1. THIẾT LẬP MÔ HÌNH VÀ CƠ SỞ DỮ LIỆU
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES_FILE = os.path.join("data", "class_names.json")
EMBEDDINGS_FILE = os.path.join("data", "class_embeddings.pt")

os.makedirs("data", exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True, device=DEVICE)
cnn_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Nạp dữ liệu hiện có (nếu có)
class_names_dict = {}
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names_dict = {int(k): v["class_name"] for k, v in mapping.items()}

class_centroids = torch.load(EMBEDDINGS_FILE, map_location=DEVICE) if os.path.exists(EMBEDDINGS_FILE) else None

def extract_embedding_with_fallback(img):
    """Trích xuất Vector 512-D từ ảnh PIL (có cơ chế Fallback nếu MTCNN không nhận diện được mặt)."""
    if img.mode != 'RGB':
        img = img.convert('RGB')

    face_tensor = mtcnn(img)
    if face_tensor is None:
        img_resized = img.resize((160, 160))
        img_np = torch.tensor(list(img_resized.getdata()), dtype=torch.float32).view(160, 160, 3)
        face_tensor = img_np.permute(2, 0, 1)
        face_tensor = (face_tensor - 127.5) / 128.0

    with torch.no_grad():
        emb = cnn_embedder(face_tensor.unsqueeze(0).to(DEVICE))
        return F.normalize(emb, p=2, dim=1)

# 2. TẢI 500 ẢNH NGẪU NHIÊN CHƯA TỪNG DÙNG QUA STREAMING

print("⚡ Đang kết nối Stream đến Hugging Face (Lấy 500 ảnh ngẫu nhiên độc lập)...")
# Bỏ qua 500 ảnh đầu để đảm bảo tập ảnh kiểm thử hoàn toàn ngẫu nhiên và unseen
hf_dataset = load_dataset("wuji3/face-recognition", split="train", streaming=True)
test_samples = list(hf_dataset.skip(1000).take(500))

print(f"✅ Đã tải thành công {len(test_samples)} ảnh ngẫu nhiên qua Stream.")

# 3. ĐĂNG KÝ LẦN LƯỢT 500 ẢNH VÀO CƠ SỞ DỮ LIỆU (CLASS ID TĂNG DẦN)
# ============================================================
print("\n🔄 BẮT ĐẦU QUY TRÌNH TỰ ĐỘNG ĐĂNG KÝ 500 KHUÔN MẶT MỚI...")

registered_records = []  # Lưu danh sách quản lý test

for i, sample in enumerate(tqdm(test_samples, desc="Đang đăng ký ảnh vào Database")):
    img = sample.get('image') or sample.get('img')
    if img is None:
        continue
    if not isinstance(img, Image.Image):
        img = Image.open(img)

    emb = extract_embedding_with_fallback(img)

    # Đặt Class ID tự động tăng dần
    current_class_id = len(class_centroids) if class_centroids is not None else 0
    assigned_name = f"Registered_Person_{current_class_id:03d}"

    # 1. Cập nhật JSON
    class_names_dict[current_class_id] = assigned_name
    formatted_json = {str(k): {"class_name": v} for k, v in class_names_dict.items()}
    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(formatted_json, f, ensure_ascii=False, indent=2)

    # 2. Cập nhật File Tensor Embeddings .pt
    if class_centroids is None:
        class_centroids = emb
    else:
        class_centroids = torch.cat([class_centroids, emb], dim=0)

    torch.save(class_centroids, EMBEDDINGS_FILE)

    registered_records.append({
        "sample_index": i,
        "assigned_class_id": current_class_id,
        "assigned_name": assigned_name,
        "img_obj": img
    })

print(f" Đã đăng ký thành công {len(registered_records)} Class ID mới vào cơ sở dữ liệu.")

print("\n BẮT ĐẦU ĐÁNH GIÁ ĐỘ CHÍNH XÁC NHẬN DIỆN VỚI CƠ SỞ DỮ LIỆU VỪA CẬP NHẬT...")

top1_correct = 0
top5_correct = 0
total_tested = len(registered_records)
eval_logs = []

for record in tqdm(registered_records, desc="Đang đối soát nhận diện"):
    img = record["img_obj"]
    expected_id = record["assigned_class_id"]
    expected_name = record["assigned_name"]

    # Trích xuất vector ảnh test
    emb = extract_embedding_with_fallback(img)

    # Đối soát Cosine với toàn bộ Centroids trong Database
    sims = F.cosine_similarity(emb, class_centroids, dim=1)
    top5_sims, top5_indices = torch.topk(sims, min(5, len(sims)))

    pred_top1_id = top5_indices[0].item()
    pred_top1_name = class_names_dict.get(pred_top1_id, f"ID_{pred_top1_id}")
    top1_score = round(top5_sims[0].item(), 4)

    top5_ids = top5_indices.tolist()

    is_top1_hit = (pred_top1_id == expected_id)
    is_top5_hit = (expected_id in top5_ids)

    if is_top1_hit:
        top1_correct += 1
    if is_top5_hit:
        top5_correct += 1

    eval_logs.append({
        "Class ID Đã Đăng Ký": expected_id,
        "Tên Đã Đăng Ký": expected_name,
        "Class ID Dự Đoán Top-1": pred_top1_id,
        "Tên Dự Đoán Top-1": pred_top1_name,
        "Cosine Similarity": top1_score,
        "Kết Quả Top-1": "✅ Đúng" if is_top1_hit else "❌ Sai",
        "Kết Quả Top-5": "✅ Đúng" if is_top5_hit else "❌ Sai"
    })

top1_acc = (top1_correct / total_tested) * 100 if total_tested > 0 else 0
top5_acc = (top5_correct / total_tested) * 100 if total_tested > 0 else 0

df_eval = pd.DataFrame(eval_logs)
output_csv = os.path.join("data", "stress_test_registered_500_results.csv")
df_eval.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\n" + "="*65)
print(" BÁO CÁO ĐỘ CHÍNH XÁC ĐĂNG KÝ & NHẬN DIỆN (STRESS TEST REPORT)")
print("="*65)
print(f" Tổng số ảnh thử nghiệm đăng ký : {total_tested} ảnh")
print(f" Số ảnh nhận diện chính xác Top-1: {top1_correct}/{total_tested}")
print(f" Độ chính xác Top-1 (Top-1 Acc)  : {top1_acc:.2f}%")
print(f" Độ chính xác Top-5 (Top-5 Acc)  : {top5_acc:.2f}%")
print(f" Điểm Cosine trung bình          : {df_eval['Cosine Similarity'].mean():.4f}")
print("-" * 65)
print(" Xem trước 10 kết quả đối soát đầu tiên:")
print(df_eval.head(10).to_string(index=False))
print("="*65)
print(f" Báo cáo chi tiết từng ảnh đã lưu tại: '{output_csv}'")