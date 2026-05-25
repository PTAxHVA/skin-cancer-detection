"""Đường dẫn, feature và mapping nhãn dùng chung."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "HAM10000_metadata.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Bỏ dx_type, dataset (leakage + không có lúc inference) và các cột id.
NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = ["sex", "localization"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_RAW = "dx"
TARGET_BINARY = "label_binary"
TARGET_MULTI = "label_multi"

# 'healthy' không nằm trong 7 loại chuẩn.
DROP_CLASSES = ["healthy"]

DX_FULL_NAME = {
    "nv": "Melanocytic nevi (nốt ruồi lành)",
    "mel": "Melanoma (u hắc tố ác tính)",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / Bowen",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

# 1 = cần lưu ý, 0 = lành tính.
MALIGNANT_GROUP = {"mel", "bcc", "akiec", "vasc"}
BINARY_LABELS = {0: "Lanh tinh (Benign)", 1: "Can luu y (Malignant-like)"}

RANDOM_STATE = 42
TEST_SIZE = 0.2
