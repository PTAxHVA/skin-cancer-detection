"""Cấu hình tập trung cho dự án Skin Cancer Detection.

Gom toàn bộ hằng số (đường dẫn, danh sách feature, mapping nhãn) về một chỗ
để code dễ đọc, dễ bảo trì và tránh "magic string" rải rác (DRY).
"""
from __future__ import annotations

from pathlib import Path

# --- Đường dẫn (tính tương đối so với gốc dự án) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "HAM10000_metadata.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Feature dùng cho mô hình ---
# Chỉ dùng đặc trưng phía bệnh nhân (có sẵn lúc người dùng nhập liệu trên demo).
# CỐ Ý loại 'dx_type' và 'dataset': chúng là metadata về quy trình thu thập dữ
# liệu, tương quan mạnh với nhãn (rò rỉ - data leakage) và KHÔNG tồn tại ở thời
# điểm dự đoán thực tế.
NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = ["sex", "localization"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Nhãn ---
TARGET_RAW = "dx"            # cột chẩn đoán gốc (mã viết tắt)
TARGET_BINARY = "label_binary"
TARGET_MULTI = "label_multi"

# Lớp 'healthy' (109 dòng) KHÔNG thuộc 7 loại chuẩn của đề bài -> loại bỏ.
DROP_CLASSES = ["healthy"]

# 7 loại tổn thương da chuẩn (HAM10000) -> tên đầy đủ (mapping dx_dict).
DX_FULL_NAME = {
    "nv": "Melanocytic nevi (nốt ruồi lành)",
    "mel": "Melanoma (u hắc tố ác tính)",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / Bowen",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

# Phân nhóm nhị phân theo đúng spec capstone:
#   Nhóm 1 (cần lưu ý / nghi ngờ): mel, bcc, akiec, vasc
#   Nhóm 0 (lành tính phổ biến):   nv, df, bkl
MALIGNANT_GROUP = {"mel", "bcc", "akiec", "vasc"}
BINARY_LABELS = {0: "Lanh tinh (Benign)", 1: "Can luu y (Malignant-like)"}

RANDOM_STATE = 42
TEST_SIZE = 0.2
