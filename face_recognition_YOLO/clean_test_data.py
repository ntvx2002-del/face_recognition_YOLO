import os
import shutil

# 1. Danh sách thư mục ảnh test cần xóa
TEST_FOLDERS = [
    os.path.join("data", "eval_200_images"),
    os.path.join("data", "eval_500_images")
]

# 2. Danh sách các file báo cáo kiểm thử cần xóa
TEST_FILES = [
    os.path.join("data", "hf_200_eval_report.csv"),
    os.path.join("data", "accuracy_200_report.csv"),
    os.path.join("data", "danh_sach_dung.csv"),
    os.path.join("data", "danh_sach_sai.csv"),
    os.path.join("data", "phan_tich_chi_tiet_200_anh.csv"),
    os.path.join("data", "stress_test_registered_200_results.csv"),
    os.path.join("data", "benchmark_summary.json"),
    os.path.join("data", "benchmark_detailed_latency.csv")
]

print("🧹 Đang dọn dẹp toàn bộ dữ liệu kiểm thử 200 & 500 ảnh...")

# Xóa các thư mục ảnh
for folder in TEST_FOLDERS:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"✅ Đã xóa thư mục ảnh test: '{folder}'")

# Xóa các file báo cáo CSV/JSON
for file_path in TEST_FILES:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Đã xóa file báo cáo: '{file_path}'")

print("\n🎉 Hoàn tất dọn dẹp! Cơ sở dữ liệu chính (class_names.json & class_embeddings.pt) vẫn được giữ an toàn.")