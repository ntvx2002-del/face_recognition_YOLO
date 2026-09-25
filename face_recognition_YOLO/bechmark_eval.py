import os
import json
import time
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
import pandas as pd
import numpy as np

# ============================================================
# 1. KHỞI TẠO MÔ HÌNH VÀ CƠ SỞ DỮ LIỆU
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES_FILE = os.path.join("data", "class_names.json")
EMBEDDINGS_FILE = os.path.join("data", "class_embeddings.pt")

os.makedirs("data", exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True, device=DEVICE)
cnn_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Nạp database danh tính local
class_names_dict = {}
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names_dict = {int(k): str(v["class_name"]).strip() for k, v in mapping.items()}

class_centroids = torch.load(EMBEDDINGS_FILE, map_location=DEVICE) if os.path.exists(EMBEDDINGS_FILE) else None

# ============================================================
# 2. CHẠY BENCHMARK TRÊN 200 ẢNH HUGGING FACE STREAMING
# ============================================================
print(f" BẮT ĐẦU CHẠY THỬ NHIỆM BENCHMARK TRÊN DEVICE: {DEVICE.type.upper()}")
print(" Đang tải 200 ảnh test qua Stream...")

hf_dataset = load_dataset("wuji3/face-recognition", split="train", streaming=True)
TOTAL_TEST = 200
test_samples = list(hf_dataset.take(TOTAL_TEST))

# Các biến lưu trữ đo đạc Latency (đơn vị: mili-giây ms)
mtcnn_times = []
embed_times = []
search_times = []
total_times = []

top1_correct = 0
top5_correct = 0
cosine_scores = []
detailed_logs = []

# Warm-up GPU/CPU (chạy thử 1 lần để tránh nhiễu khởi tạo)
if len(test_samples) > 0:
    dummy_img = Image.new('RGB', (160, 160))
    _ = cnn_embedder(mtcnn(dummy_img).unsqueeze(0).to(DEVICE) if mtcnn(dummy_img) is not None else torch.randn(1, 3, 160, 160).to(DEVICE))

print("\n Đang tiến hành đo đạc thời gian tính toán và độ chính xác...")

for i, sample in enumerate(tqdm(test_samples, desc="Benchmarking")):
    img = sample.get('image') or sample.get('img')
    if img is None:
        continue
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    gt_label_raw = sample.get('label') or sample.get('name') or sample.get('class_name') or f"class_{sample.get('label_id', i)}"
    gt_label_str = str(gt_label_raw).strip().lower()

    t0 = time.perf_counter()

    # --- ĐO GIAI ĐOẠN 1: MTCNN Alignment & Bounding Box ---
    t_mtcnn_start = time.perf_counter()
    face_tensor = mtcnn(img)
    if face_tensor is None:
        img_resized = img.resize((160, 160))
        img_np = torch.tensor(list(img_resized.getdata()), dtype=torch.float32).view(160, 160, 3)
        face_tensor = img_np.permute(2, 0, 1)
        face_tensor = (face_tensor - 127.5) / 128.0
    t_mtcnn_end = time.perf_counter()

    # --- ĐO GIAI ĐOẠN 2: FaceNet Feature Extraction ---
    t_embed_start = time.perf_counter()
    with torch.no_grad():
        emb = cnn_embedder(face_tensor.unsqueeze(0).to(DEVICE))
        emb = F.normalize(emb, p=2, dim=1)
    t_embed_end = time.perf_counter()

    # --- ĐO GIAI ĐOẠN 3: Cosine Similarity Matching ---
    t_search_start = time.perf_counter()
    if class_centroids is not None and len(class_centroids) > 0:
        sims = F.cosine_similarity(emb, class_centroids, dim=1)
        top5_sims, top5_indices = torch.topk(sims, min(5, len(sims)))
        pred_top1_idx = top5_indices[0].item()
        score = top5_sims[0].item()
        top5_ids = top5_indices.tolist()
    else:
        pred_top1_idx = 0
        score = 0.0
        top5_ids = []
    t_search_end = time.perf_counter()

    t1 = time.perf_counter()

    # Tính latency từng công đoạn (ms)
    d_mtcnn = (t_mtcnn_end - t_mtcnn_start) * 1000
    d_embed = (t_embed_end - t_embed_start) * 1000
    d_search = (t_search_end - t_search_start) * 1000
    d_total = (t1 - t0) * 1000

    mtcnn_times.append(d_mtcnn)
    embed_times.append(d_embed)
    search_times.append(d_search)
    total_times.append(d_total)
    cosine_scores.append(score)

    pred_top1_name = class_names_dict.get(pred_top1_idx, f"id_{pred_top1_idx}")
    is_top1 = (gt_label_str in pred_top1_name.lower()) or (pred_top1_name.lower() in gt_label_str) or (str(pred_top1_idx) == gt_label_str)
    is_top5 = any((gt_label_str in class_names_dict.get(idx, "").lower()) for idx in top5_ids)

    if is_top1:
        top1_correct += 1
    if is_top5:
        top5_correct += 1

    detailed_logs.append({
        "Sample_ID": i + 1,
        "Ground_Truth": gt_label_raw,
        "Predicted_Top1": pred_top1_name,
        "Cosine_Score": round(score, 4),
        "MTCNN_Latency_ms": round(d_mtcnn, 2),
        "FaceNet_Latency_ms": round(d_embed, 2),
        "Matching_Latency_ms": round(d_search, 2),
        "Total_Latency_ms": round(d_total, 2)
    })

# ============================================================
# 3. TÍNH TOÁN CÁC CHỈ SỐ BENCHMARK TỔNG HỢP
# ============================================================
n_samples = len(detailed_logs)
top1_acc = (top1_correct / n_samples) * 100 if n_samples > 0 else 0
top5_acc = (top5_correct / n_samples) * 100 if n_samples > 0 else 0

avg_mtcnn = np.mean(mtcnn_times)
avg_embed = np.mean(embed_times)
avg_search = np.mean(search_times)
avg_total = np.mean(total_times)
fps = 1000.0 / avg_total if avg_total > 0 else 0

# Tính kích thước RAM / Lưu trữ
total_identities = len(class_centroids) if class_centroids is not None else 0
embedding_size_bytes = 512 * 4  # 512 Float32 = 2048 bytes (2 KB)

benchmark_report = {
    "device_hardware": str(DEVICE).upper(),
    "total_test_samples": n_samples,
    "accuracy_metrics": {
        "top1_accuracy_pct": round(top1_acc, 2),
        "top5_accuracy_pct": round(top5_acc, 2),
        "mean_cosine_score": round(float(np.mean(cosine_scores)), 4),
        "std_cosine_score": round(float(np.std(cosine_scores)), 4)
    },
    "latency_metrics_ms": {
        "avg_mtcnn_alignment_ms": round(float(avg_mtcnn), 2),
        "avg_facenet_embedding_ms": round(float(avg_embed), 2),
        "avg_vector_search_ms": round(float(avg_search), 2),
        "avg_total_pipeline_ms": round(float(avg_total), 2),
        "throughput_fps": round(float(fps), 2)
    },
    "memory_metrics": {
        "registered_identities_count": total_identities,
        "size_per_identity_bytes": embedding_size_bytes,
        "total_embeddings_file_kb": round((total_identities * embedding_size_bytes) / 1024, 2)
    }
}

# ============================================================
# 4. IN BÁO CÁO BENCHMARK VÀ XUẤT FILE
# ============================================================
print("\n" + "="*70)
print(" BÁO CÁO ĐÁNH GIÁ BENCHMARK HỆ THỐNG (SYSTEM BENCHMARK REPORT)")
print("="*70)
print(f" Thiết bị phần cứng suy luận  : {benchmark_report['device_hardware']}")
print(f" Mẫu ảnh thử nghiệm          : {n_samples} ảnh")
print(f" Số danh tính trong Database  : {total_identities} identities")
print("-" * 70)
print(" 1. CHỈ SỐ ĐỘ CHÍNH XÁC (ACCURACY METRICS):")
print(f"   - Độ chính xác Top-1 Acc     : {top1_acc:.2f}%")
print(f"   - Độ chính xác Top-5 Acc     : {top5_acc:.2f}%")
print(f"   - Điểm Cosine trung bình    : {benchmark_report['accuracy_metrics']['mean_cosine_score']}")
print("-" * 70)
print(" 2. CHỈ SỐ THỜI GIAN XỬ LÝ & FPS (LATENCY & THROUGHPUT):")
print(f"   - MTCNN Face Alignment       : {avg_mtcnn:.2f} ms")
print(f"   - FaceNet Feature Extraction : {avg_embed:.2f} ms")
print(f"   - Vector Search Matching     : {avg_search:.2f} ms")
print(f"   - Tổng thời gian 1 khung hình: {avg_total:.2f} ms")
print(f"   - Tốc độ xử lý thực tế (FPS)  : {fps:.2f} FPS")
print("-" * 70)
print(" 3. BỘ NHỚ VÀ DUNG LƯỢNG LƯU TRỮ (RESOURCE FOOTPRINT):")
print(f"   - Kích thước 1 Vector 512-D  : {embedding_size_bytes} Bytes (~2 KB)")
print(f"   - Dung lượng file .pt        : {benchmark_report['memory_metrics']['total_embeddings_file_kb']} KB")
print("="*70)

# Lưu báo cáo JSON
json_path = os.path.join("data", "benchmark_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_report, f, ensure_ascii=False, indent=2)

# Lưu báo cáo CSV chi tiết từng khung hình
csv_path = os.path.join("data", "benchmark_detailed_latency.csv")
pd.DataFrame(detailed_logs).to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\n Đã lưu báo cáo tổng hợp JSON tại  : '{json_path}'")
print(f" Đã lưu log Latency chi tiết CSV tại: '{csv_path}'")