import psycopg2
import torch
import torch.nn.functional as F

# Cấu hình kết nối PostgreSQL
DB_CONFIG = {
    "dbname": "facenet_db",
    "user": "postgres",
    "password": "vu123456",  # Đổi thành mật khẩu PostgreSQL thực tế của bạn
    "host": "localhost",
    "port": "5432"
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def load_database(device="cpu"):
    """Nạp toàn bộ Class ID, Tên và Vector Centroid từ PostgreSQL vào RAM/GPU."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT class_id, person_name, embedding FROM face_identities ORDER BY class_id ASC;")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        class_names_dict = {}
        embeddings_list = []

        for row in rows:
            c_id, name, emb_list = row
            class_names_dict[c_id] = name
            embeddings_list.append(emb_list)

        if len(embeddings_list) > 0:
            class_centroids = torch.tensor(embeddings_list, dtype=torch.float32).to(device)
            class_centroids = F.normalize(class_centroids, p=2, dim=1)
        else:
            class_centroids = None

        return class_names_dict, class_centroids
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu PostgreSQL: {e}")
        return {}, None

def save_new_face(person_name, embedding_tensor):
    """Đăng ký khuôn mặt mới vào PostgreSQL."""
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Tự động tính Class ID tiếp theo
        cur.execute("SELECT COALESCE(MAX(class_id), -1) + 1 FROM face_identities;")
        new_class_id = cur.fetchone()[0]

        emb_list = embedding_tensor.squeeze().detach().cpu().numpy().tolist()

        cur.execute("""
            INSERT INTO face_identities (class_id, person_name, embedding)
            VALUES (%s, %s, %s);
        """, (new_class_id, person_name, emb_list))

        conn.commit()
        cur.close()
        conn.close()
        return new_class_id
    except Exception as e:
        print(f"❌ Lỗi lưu dữ liệu PostgreSQL: {e}")
        return None

def update_person_name(class_id, new_name):
    """Cập nhật tên danh tính trong PostgreSQL."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE face_identities SET person_name = %s, updated_at = CURRENT_TIMESTAMP WHERE class_id = %s;", (new_name, class_id))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Lỗi cập nhật tên PostgreSQL: {e}")
        return False

def delete_face_identity(class_id):
    """Xóa danh tính và sắp xếp lại chỉ số Class ID liên tục."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM face_identities WHERE class_id = %s;", (class_id,))

        # Đánh lại chỉ số class_id liên tục 0..N-1
        cur.execute("""
            WITH reordered AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY class_id) - 1 AS new_class_id
                FROM face_identities
            )
            UPDATE face_identities f
            SET class_id = r.new_class_id
            FROM reordered r
            WHERE f.id = r.id;
        """)
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Lỗi xóa danh tính PostgreSQL: {e}")
        return False

def get_all_identities_df():
    """Lấy danh sách toàn bộ danh tính dạng List để hiển thị trên bảng Gradio."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT class_id, person_name, created_at FROM face_identities ORDER BY class_id ASC;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [[r[0], r[1], str(r[2])] for r in rows]
    except Exception as e:
        print(f"❌ Lỗi lấy danh sách PostgreSQL: {e}")
        return []