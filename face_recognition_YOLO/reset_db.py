import os

files_to_remove = [
    os.path.join("data", "class_names.json"),
    os.path.join("data", "class_embeddings.pt")
]

for f in files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        print(f"✅ Đã xóa file: {f}")

print("🎉 Database đã được xóa sạch! Lần tới chạy app.py hoặc camera_app.py, hệ thống sẽ tự khởi tạo lại cơ sở dữ liệu mới từ ID 0.")