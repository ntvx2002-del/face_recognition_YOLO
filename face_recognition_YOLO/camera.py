import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import InceptionResnetV1, MTCNN
# Import các hàm tương tác PostgreSQL
from db_manager import load_database, save_new_face

# ============================================================
# 1. KHỞI TẠO MÔ HÌNH VÀ KẾT NỐI POSTGRESQL
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACES_DIR = os.path.join("data", "registered_faces")
os.makedirs(FACES_DIR, exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True, device=DEVICE)
cnn_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Nạp dữ liệu từ PostgreSQL
class_names_dict, class_centroids = load_database(device=DEVICE)

SIMILARITY_THRESHOLD = 0.45 # Ngưỡng phân biệt người quen vs người lạ


def draw_vietnamese_text(img_bgr, text, position, color=(0, 255, 0), font_size=20):
    """Vẽ chữ tiếng Việt có dấu lên khung hình OpenCV."""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def register_face_to_postgres(new_emb, person_name, cropped_pil):
    """Lưu danh tính vào PostgreSQL và lưu ảnh đại diện crop ra đĩa."""
    global class_names_dict, class_centroids
    final_name = person_name.strip()

    # 1. Lưu trực tiếp vào CSDL PostgreSQL
    new_id = save_new_face(person_name=final_name, embedding_tensor=new_emb)

    if new_id is not None:
        # 2. Nạp lại CSDL từ PostgreSQL lên RAM để cập nhật tức thì
        class_names_dict, class_centroids = load_database(device=DEVICE)

        # 3. Lưu ảnh crop để đối soát (tùy chọn)
        img_path = os.path.join(FACES_DIR, f"ID_{new_id:02d}_{final_name}.jpg")
        cropped_pil.save(img_path)

        print(f"\n✅ ĐÃ LƯU THÀNH CÔNG: '{final_name}' (Class ID #{new_id}) vào PostgreSQL!")
        return True
    else:
        print("\n❌ LỖI: Không thể ghi dữ liệu vào PostgreSQL!")
        return False


def main():
    global class_centroids, class_names_dict
    cap = cv2.VideoCapture(0)  # Mở Camera

    if not cap.isOpened():
        print("❌ Không thể mở Webcam!")
        return

    print("\n" + "="*60)
    print(" 🚀 HỆ THỐNG QUÉT KHUÔN MẶT TRỰC TIẾP (POSTGRESQL REAL-TIME)")
    print(" - Nhấn phím 'r' trên bàn phím để ĐĂNG KÝ khuôn mặt đang quét.")
    print(" - Nhấn phím 'q' để THOÁT chương trình.")
    print("="*60 + "\n")

    current_crop_pil = None
    current_emb = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lật gương khung hình camera
        frame = cv2.flip(frame, 1)

        # Chuyển BGR (OpenCV) sang RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Phát hiện vị trí khuôn mặt
        boxes, probs = mtcnn.detect(pil_img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]

                # Crop mặt & trích xuất Vector
                face_tensor = mtcnn(pil_img)
                if face_tensor is None:
                    # Fallback resize thủ công khi zoom quá sát
                    img_resized = pil_img.resize((160, 160))
                    img_np = torch.tensor(list(img_resized.getdata()), dtype=torch.float32).view(160, 160, 3)
                    face_tensor = img_np.permute(2, 0, 1)
                    face_tensor = (face_tensor - 127.5) / 128.0

                # Chuyển Tensor ra PIL Image để lưu file ảnh đại diện
                crop_np = face_tensor.cpu().numpy().transpose(1, 2, 0)
                crop_np = ((crop_np - crop_np.min()) / (crop_np.max() - crop_np.min()) * 255).astype(np.uint8)
                current_crop_pil = Image.fromarray(crop_np)

                with torch.no_grad():
                    emb = cnn_embedder(face_tensor.unsqueeze(0).to(DEVICE))
                    current_emb = F.normalize(emb, p=2, dim=1)

                # Đối soát Cosine với PostgreSQL Centroids
                display_name = "Khuon mat moi (Nhan 'r' de luu)"
                box_color = (0, 0, 255)  # Đỏ cho người lạ

                if class_centroids is not None and len(class_centroids) > 0:
                    sims = F.cosine_similarity(current_emb, class_centroids, dim=1)
                    top_sim, top_idx = torch.max(sims, dim=0)
                    score = top_sim.item()

                    if score >= SIMILARITY_THRESHOLD:
                        match_name = class_names_dict.get(top_idx.item(), f"ID_{top_idx.item()}")
                        pct = max(0.0, min(100.0, (score - 0.20) / (0.70 - 0.20) * 100))
                        display_name = f"{match_name} ({pct:.1f}%)"
                        box_color = (0, 255, 0)  # Xanh lá cho người quen

                # Vẽ khung Bounding Box & Tên người
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                frame = draw_vietnamese_text(frame, display_name, (x1, max(0, y1 - 30)), color=box_color, font_size=22)

        cv2.imshow("Camera Face Scanner - PostgreSQL", frame)

        key = cv2.waitKey(1) & 0xFF
        # PHÍM 'r': ĐĂNG KÝ KHUÔN MẶT MỚI VÀO POSTGRESQL
        if key == ord('r'):
            if current_emb is not None and current_crop_pil is not None:
                print("\n📸 BẠN VỪA BẤM LƯU KHUÔN MẶT VÀO POSTGRESQL!")
                name_input = input(" ✏️ Nhập Tên cho khuôn mặt này: ")
                if name_input.strip():
                    register_face_to_postgres(current_emb, name_input, current_crop_pil)
                else:
                    print(" ⚠️ Đã hủy lưu do chưa nhập tên.")
            else:
                print(" ⚠️ Chưa phát hiện khuôn mặt nào trong khung hình!")

        # PHÍM 'q': THOÁT
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()