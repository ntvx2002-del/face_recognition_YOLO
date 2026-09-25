import io
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import InceptionResnetV1, MTCNN
import gradio as gr

# Import các hàm tương tác CSDL từ db_manager.py
from db_manager import (
    load_database,
    save_new_face,
    update_person_name,
    delete_face_identity,
    get_all_identities_df
)

# ============================================================
# 1. KHỞI TẠO MÔ HÌNH VÀ KẾT NỐI POSTGRESQL
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_THRESHOLD = 0.45
FACES_DIR = os.path.join("data", "registered_faces")
os.makedirs(FACES_DIR, exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True, device=DEVICE)
cnn_embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Nạp dữ liệu ban đầu từ PostgreSQL lên RAM
class_names_dict, class_centroids = load_database(device=DEVICE)


# ============================================================
# 2. HÀM BỔ TRỢ & TRÍCH XUẤT VECTOR
# ============================================================
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


def extract_embedding(img_input):
    """Trích xuất Vector 512-D với cơ chế Fallback thủ công."""
    if img_input is None:
        return None

    if not isinstance(img_input, Image.Image):
        img_input = Image.fromarray(img_input)

    if img_input.mode != 'RGB':
        img_input = img_input.convert('RGB')

    face_tensor = mtcnn(img_input)

    # Fallback khi MTCNN không bắt được mặt do zoom quá sát
    if face_tensor is None:
        img_resized = img_input.resize((160, 160))
        img_np = torch.tensor(list(img_resized.getdata()), dtype=torch.float32).view(160, 160, 3)
        face_tensor = img_np.permute(2, 0, 1)
        face_tensor = (face_tensor - 127.5) / 128.0

    with torch.no_grad():
        emb = cnn_embedder(face_tensor.unsqueeze(0).to(DEVICE))
        return F.normalize(emb, p=2, dim=1)


# ============================================================
# 3. CHỨC NĂNG 1: CAMERA REAL-TIME QUÉT TRỰC TIẾP (OPENCV)
# ============================================================
def start_live_camera():
    """Kích hoạt luồng Camera OpenCV trực tiếp, nhấn 'r' để đăng ký, 'q' để thoát."""
    global class_centroids, class_names_dict
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return "Lỗi: Không thể truy cập Webcam!"

    print("\n" + "="*60)
    print(" ĐÃ KÍCH HOẠT CAMERA QUÉT TRỰC TIẾP")
    print(" - Nhấn phím 'r' trên bàn phím để ĐĂNG KÝ khuôn mặt.")
    print(" - Nhấn phím 'q' để ĐÓNG CAMERA.")
    print("="*60 + "\n")

    current_crop_pil = None
    current_emb = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        boxes, probs = mtcnn.detect(pil_img)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]

                face_tensor = mtcnn(pil_img)
                if face_tensor is None:
                    img_resized = pil_img.resize((160, 160))
                    img_np = torch.tensor(list(img_resized.getdata()), dtype=torch.float32).view(160, 160, 3)
                    face_tensor = img_np.permute(2, 0, 1)
                    face_tensor = (face_tensor - 127.5) / 128.0

                crop_np = face_tensor.cpu().numpy().transpose(1, 2, 0)
                crop_np = ((crop_np - crop_np.min()) / (crop_np.max() - crop_np.min()) * 255).astype(np.uint8)
                current_crop_pil = Image.fromarray(crop_np)

                with torch.no_grad():
                    emb = cnn_embedder(face_tensor.unsqueeze(0).to(DEVICE))
                    current_emb = F.normalize(emb, p=2, dim=1)

                display_name = "Khuon mat moi (Nhan 'r' de luu)"
                box_color = (0, 0, 255)

                if class_centroids is not None and len(class_centroids) > 0:
                    sims = F.cosine_similarity(current_emb, class_centroids, dim=1)
                    top_sim, top_idx = torch.max(sims, dim=0)
                    score = top_sim.item()

                    if score >= SIMILARITY_THRESHOLD:
                        match_name = class_names_dict.get(top_idx.item(), f"ID_{top_idx.item()}")
                        pct = max(0.0, min(100.0, (score - 0.20) / (0.70 - 0.20) * 100))
                        display_name = f"{match_name} ({pct:.1f}%)"
                        box_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                frame = draw_vietnamese_text(frame, display_name, (x1, max(0, y1 - 30)), color=box_color, font_size=22)

        cv2.imshow("Camera Real-Time Scanner - Nhan 'r' de Luu / 'q' de Thoat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if current_emb is not None and current_crop_pil is not None:
                print("\n BẠN VỪA BẤM LƯU KHUÔN MẶT VÀO POSTGRESQL!")
                name_input = input("  Nhập Tên cho khuôn mặt này: ")
                if name_input.strip():
                    final_name = name_input.strip()
                    new_id = save_new_face(person_name=final_name, embedding_tensor=current_emb)

                    if new_id is not None:
                        class_names_dict, class_centroids = load_database(device=DEVICE)
                        img_path = os.path.join(FACES_DIR, f"ID_{new_id:02d}_{final_name}.jpg")
                        current_crop_pil.save(img_path)
                        print(f" ĐÃ LƯU THÀNH CÔNG: '{final_name}' (Class ID #{new_id}) VÀO POSTGRESQL!")
                else:
                    print("  Đã hủy lưu do chưa nhập tên.")
            else:
                print(" Chưa phát hiện khuôn mặt nào trong khung hình!")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return " Đã đóng phiên làm việc Camera Real-Time!"


# ============================================================
# 4. CHỨC NĂNG 2: PHÂN TÍCH VÀ ĐĂNG KÝ QUA ẢNH TẢI LÊN
# ============================================================
def analyze_and_identify(uploaded_image, custom_name):
    global class_centroids, class_names_dict

    if uploaded_image is None:
        return None, " Vui lòng tải ảnh lên!", None, None, None

    w, h = uploaded_image.size
    img_byte_arr = io.BytesIO()
    uploaded_image.save(img_byte_arr, format=uploaded_image.format or 'JPEG')
    file_size_kb = len(img_byte_arr.getvalue()) / 1024

    face_tensor = mtcnn(uploaded_image)

    if face_tensor is None:
        img_resized = uploaded_image.convert("RGB").resize((160, 160))
        img_np = np.array(img_resized).astype(np.float32)
        face_tensor = torch.tensor(img_np).permute(2, 0, 1)
        face_tensor = (face_tensor - 127.5) / 128.0

    crop_np = face_tensor.cpu().numpy().transpose(1, 2, 0)
    crop_np = ((crop_np - crop_np.min()) / (crop_np.max() - crop_np.min()) * 255).astype(np.uint8)
    cropped_pil = Image.fromarray(crop_np)

    with torch.no_grad():
        input_batch = face_tensor.unsqueeze(0).to(DEVICE)
        img_embedding = cnn_embedder(input_batch)
        img_embedding = F.normalize(img_embedding, p=2, dim=1)

    if class_centroids is not None and len(class_centroids) > 0:
        similarities = F.cosine_similarity(img_embedding, class_centroids, dim=1)
        top5_sims, top5_indices = torch.topk(similarities, min(5, len(similarities)))
        raw_cosine = top5_sims[0].item()
        top1_id = top5_indices[0].item()
    else:
        raw_cosine = 0.0
        top1_id = -1

    if raw_cosine >= SIMILARITY_THRESHOLD:
        match_pct = max(0.0, min(100.0, (raw_cosine - 0.20) / (0.70 - 0.20) * 100))
        top1_name = class_names_dict.get(top1_id, f"ID_{top1_id}")

        main_pred_html = f"""
        <div style="background-color: #d4edda; border-left: 5px solid #28a745; padding: 15px; border-radius: 4px;">
            <h2 style="margin:0; color: #155724;">👤 Danh Tính: <span>{top1_name}</span></h2>
            <p style="margin: 5px 0 0 0; font-size: 16px;">Độ trùng khớp sinh trắc học: <b>{match_pct:.2f}%</b> (Cosine: {raw_cosine:.3f})</p>
        </div>
        """
    else:
        if not custom_name or not custom_name.strip():
            main_pred_html = f"""
            <div style="background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 4px;">
                <h2 style="margin:0; color: #856404;"> PHÁT HIỆN KHUÔN MẶT MỚI</h2>
                <p style="margin: 5px 0 0 0; font-size: 15px;">Vui lòng <b>nhập Tên vào ô bên trái</b> và nhấn lại nút <b>Phân Tích & Dự Đoán</b> để đăng ký vào PostgreSQL!</p>
            </div>
            """
        else:
            new_id_idx = save_new_face(person_name=custom_name.strip(), embedding_tensor=img_embedding)
            if new_id_idx is not None:
                class_names_dict, class_centroids = load_database(device=DEVICE)
                registered_name = custom_name.strip()

                main_pred_html = f"""
                <div style="background-color: #cce5ff; border-left: 5px solid #004085; padding: 15px; border-radius: 4px;">
                    <h2 style="margin:0; color: #004085;">🎉 ĐÃ ĐĂNG KÝ THÀNH CÔNG KHUÔN MẶT MỚI</h2>
                    <p style="margin: 5px 0 0 0; font-size: 16px;">Họ và Tên: <b>{registered_name}</b> (Class ID: #{new_id_idx})</p>
                </div>
                """
            else:
                main_pred_html = f"""
                <div style="background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 4px;">
                    <h2 style="margin:0; color: #721c24;"> LỖI ĐĂNG KÝ POSTGRESQL</h2>
                </div>
                """

    top5_data = []
    confidence_chart = {}
    if class_centroids is not None and len(class_centroids) > 0:
        similarities = F.cosine_similarity(img_embedding, class_centroids, dim=1)
        top5_sims, top5_indices = torch.topk(similarities, min(5, len(similarities)))
        for rank, (idx, score) in enumerate(zip(top5_indices.tolist(), top5_sims.tolist()), start=1):
            name = class_names_dict.get(idx, f"ID_{idx}")
            pct = max(0.0, min(100.0, (score - 0.20) / (0.70 - 0.20) * 100))
            top5_data.append({
                "Thứ hạng": f"Top {rank}",
                "Class ID": idx,
                "Danh tính": name,
                "Độ trùng khớp (%)": f"{pct:.2f}%"
            })
            confidence_chart[name] = float(score)

    tech_specs_html = f"""
    <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 12px; border-radius: 6px;">
        <h3 style="margin: 0; color: #343a40;">📐 Thông Số Kỹ Thuật</h3>
        <ul style="margin: 5px 0 0 0; line-height: 1.6;">
            <li><b>Kích thước gốc:</b> {w} x {h} px ({file_size_kb:.2f} KB)</li>
            <li><b>Cosine Similarity:</b> {raw_cosine:.4f}</li>
            <li><b>Lưu trữ CSDL:</b> PostgreSQL (Table face_identities)</li>
        </ul>
    </div>
    """

    return cropped_pil, tech_specs_html, main_pred_html, pd.DataFrame(top5_data), confidence_chart


# ============================================================
# 5. CHỨC NĂNG 3: BẢO TRÌ & QUẢN LÝ CSDL (TAB 2)
# ============================================================
def get_identities_dataframe():
    data = get_all_identities_df()
    if not data:
        return pd.DataFrame(columns=["Class ID", "Tên danh tính", "Ngày tạo"])
    return pd.DataFrame(data, columns=["Class ID", "Tên danh tính", "Ngày tạo"])


def edit_identity_name(target_id, new_name):
    global class_names_dict, class_centroids
    if target_id is None or not new_name or not new_name.strip():
        return get_identities_dataframe(), " Vui lòng nhập Class ID hợp lệ và Tên mới!"

    target_id = int(target_id)
    success = update_person_name(target_id, new_name.strip())

    if success:
        class_names_dict, class_centroids = load_database(device=DEVICE)
        return get_identities_dataframe(), f" Đã đổi tên Class ID #{target_id} thành: '{new_name.strip()}'"
    return get_identities_dataframe(), f" Không tìm thấy Class ID #{target_id}!"


def delete_identity(target_id):
    global class_names_dict, class_centroids
    if target_id is None:
        return get_identities_dataframe(), " Vui lòng nhập Class ID cần xóa!"

    target_id = int(target_id)
    old_name = class_names_dict.get(target_id, f"ID_{target_id}")

    success = delete_face_identity(target_id)

    if success:
        class_names_dict, class_centroids = load_database(device=DEVICE)
        return get_identities_dataframe(), f" Đã xóa thành công danh tính '{old_name}' (Class ID #{target_id}) khỏi PostgreSQL!"
    return get_identities_dataframe(), f" Lỗi khi xóa Class ID #{target_id}!"


# ============================================================
# 6. GIAO DIỆN HỢP NHẤT GRADIO
# ============================================================
with gr.Blocks(title="Hệ Thống Nhận Diện Khuôn Mặt - PostgreSQL") as demo:
    gr.Markdown("#  Hệ Thống Nhận Diện & Quản Lý Danh Tính Khuôn Mặt (PostgreSQL)")

    with gr.Tabs():
        # TAB 1: NHẬN DIỆN & ĐĂNG KÝ
        with gr.Tab(" Nhận Diện &  Đăng Ký"):
            # NÚT BẤM KÍCH HOẠT CAMERA QUÉT TRỰC TIẾP
            with gr.Row():
                btn_live_cam = gr.Button(" MỞ CAMERA QUÉT TRỰC TIẾP REAL-TIME (OPENCV)", variant="stop", size="lg")
            txt_cam_status = gr.Textbox(label="Trạng thái phiên làm việc Camera", interactive=False)

            gr.Markdown("---")
            gr.Markdown("###  Hoặc phân tích & đăng ký từ File Ảnh tải lên:")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Tải ảnh khuôn mặt")
                    name_input = gr.Textbox(
                        label="Tên đăng ký người mới (Nhập khi muốn đăng ký người mới)",
                        placeholder="Ví dụ: Nguyễn Văn A..."
                    )
                    btn_run = gr.Button("🔍 Phân Tích & Dự Đoán", variant="primary")
                    cropped_output = gr.Image(type="pil", label="Khuôn mặt đã crop (MTCNN)")
                    tech_output = gr.HTML()
                with gr.Column(scale=1):
                    main_output = gr.HTML()
                    chart_output = gr.Label(num_top_classes=5, label="Phân phối độ tương đồng Top-5")
                    table_output = gr.Dataframe(label="Bảng kết quả đối soát chi tiết")

            btn_live_cam.click(fn=start_live_camera, inputs=[], outputs=[txt_cam_status])
            btn_run.click(
                fn=analyze_and_identify,
                inputs=[image_input, name_input],
                outputs=[cropped_output, tech_output, main_output, table_output, chart_output]
            )

        # TAB 2: QUẢN LÝ DANH TÍNH
        with gr.Tab(" Quản Lý Danh Tính"):
            gr.Markdown("###  Danh Sách Tất Cả Các Class Đã Đăng Ký Trong PostgreSQL")
            list_table = gr.Dataframe(value=get_identities_dataframe, label="Danh sách cơ sở dữ liệu")
            btn_refresh = gr.Button(" Tải lại danh sách")

            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("####  Đổi Tên Danh Tính")
                    edit_id_input = gr.Number(label="Class ID cần đổi tên", precision=0)
                    edit_name_input = gr.Textbox(label="Tên mới", placeholder="Nhập tên mới...")
                    btn_edit = gr.Button(" Lưu Tên Mới", variant="secondary")

                with gr.Column():
                    gr.Markdown("####  Xóa Danh Tính")
                    delete_id_input = gr.Number(label="Class ID cần xóa", precision=0)
                    btn_delete = gr.Button(" Xóa Danh Tính", variant="stop")

            status_output = gr.Markdown("")

            btn_refresh.click(fn=get_identities_dataframe, inputs=[], outputs=[list_table])
            btn_edit.click(
                fn=edit_identity_name,
                inputs=[edit_id_input, edit_name_input],
                outputs=[list_table, status_output]
            )
            btn_delete.click(
                fn=delete_identity,
                inputs=[delete_id_input],
                outputs=[list_table, status_output]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)