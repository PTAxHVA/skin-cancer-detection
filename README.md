---
title: Skin Cancer Detection
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
license: mit
---

# 🔬 Skin Cancer Detection — Capstone Project (Machine Learning)

Phân loại tổn thương da dựa trên **metadata bệnh nhân** (bộ dữ liệu **HAM10000** của ISIC),
xây dựng **pipeline Machine Learning hoàn chỉnh** với Scikit-learn: tiền xử lý → huấn luyện
SVM → đánh giá → cross-validation & tuning → demo Streamlit.

> Khoá **Machine Learning & Deep Learning — Cybersoft** · Capstone Buổi 10.

---

## 🎯 Bài toán

| Bài toán | Mô tả | Nhãn |
|----------|-------|------|
| **Binary** | Lành tính vs Cần lưu ý | `0` = nv, df, bkl · `1` = mel, bcc, akiec, vasc |
| **Multi-class** | 7 loại tổn thương da | nv, mel, bkl, bcc, akiec, vasc, df |

Dữ liệu dùng `HAM10000_metadata.csv` (~10k mẫu). Đặc trưng dùng cho mô hình: **`age`, `sex`,
`localization`** — đây là thông tin phía bệnh nhân có sẵn lúc dự đoán thực tế.

> ⚠️ **Quyết định thiết kế quan trọng:** loại bỏ `dx_type` và `dataset` vì chúng **rò rỉ nhãn**
> (data leakage) và không tồn tại ở thời điểm inference; loại lớp `healthy` (109 mẫu) vì không
> thuộc 7 loại chuẩn của đề.

---

## 📊 Kết quả

Mô hình chính: **SVM (RBF)** + `StandardScaler`, tinh chỉnh bằng `GridSearchCV` (5-fold),
`class_weight='balanced'`.

| Bài toán | Best params | CV macro-F1 | Test accuracy | Test macro-F1 |
|----------|-------------|-------------|---------------|---------------|
| Binary | `C=10, gamma=0.01` | 0.651 | 0.714 | **0.633** |
| Multi-class | `C=10, gamma='scale'` | 0.195 | 0.323 | 0.182 |

**Phân tích trung thực:**
- **Binary** dùng tốt như công cụ **sàng lọc**: phát hiện ~58% ca "cần lưu ý" (recall),
  trong khi baseline đoán toàn "lành tính" = 0%. Mô hình đánh đổi accuracy lấy recall lớp
  hiếm — đúng ưu tiên y khoa.
- **Multi-class** macro-F1 thấp → minh chứng **metadata nhân khẩu học không đủ** tách 7 loại
  bệnh. Đây chính là lý do hướng mở rộng dùng **CNN trên ảnh thô**.
- Trên **macro-F1** (chỉ số đúng cho dữ liệu mất cân bằng), SVM cạnh tranh ngang RandomForest
  và vượt XGBoost/KNN — vốn chỉ tối ưu accuracy bằng cách thiên về lớp đa số (`nv` ~66%).

Biểu đồ confusion matrix, so sánh mô hình, phân bố lớp: xem `reports/figures/`.

---

## 🗂️ Cấu trúc dự án

```
skin-cancer-detection/
├── app.py                       # Demo Streamlit (deploy)
├── requirements.txt
├── README.md  /  DEPLOY.md  /  YOUTUBE_SCRIPT.md
├── data/
│   └── HAM10000_metadata.csv
├── src/                         # Mã nguồn module hoá (PEP 8, <200 dòng/file)
│   ├── config.py                # Hằng số, đường dẫn, mapping nhãn
│   ├── data_preprocessing.py    # Load, impute, encode, scale (Preprocessor)
│   ├── train.py                 # Huấn luyện + tuning + lưu artifact
│   ├── evaluation.py            # Confusion matrix, report, biểu đồ
│   └── predict.py               # Inference cho demo
├── notebooks/
│   └── skin_cancer_detection.ipynb   # EDA → train → eval (đã chạy, có output)
├── models/                      # *.joblib (model + preprocessor + encoder)
└── reports/
    ├── metrics.json
    └── figures/                 # *.png
```

---

## 🚀 Chạy local

```bash
# 1. Cài thư viện (khuyến nghị virtualenv/conda)
pip install -r requirements.txt

# 2. (Tuỳ chọn) huấn luyện lại — sinh models/ + reports/
python -m src.train

# 3. Chạy demo
streamlit run app.py        # mở http://localhost:8501
```

Notebook: mở `notebooks/skin_cancer_detection.ipynb` bằng Jupyter để xem toàn bộ phân tích.

---

## 🧰 Công nghệ

`Python 3` · `pandas` · `numpy` · `scikit-learn` (SVC, SimpleImputer, LabelEncoder,
StandardScaler, train_test_split, GridSearchCV) · `xgboost` · `matplotlib` · `seaborn`
· `streamlit` · `joblib`.

---

## 🌐 Triển khai (Deploy)

Xem hướng dẫn chi tiết từng bước trong **[`DEPLOY.md`](DEPLOY.md)** — gồm GitHub + Hugging
Face Spaces + Streamlit Community Cloud.

---

## 🔭 Hướng mở rộng

1. **CNN trên ảnh thô** (ResNet/EfficientNet + transfer learning) — giải pháp thực sự.
2. **API** với FastAPI/Flask + web UI.
3. Mô hình **multimodal**: kết hợp metadata + đặc trưng ảnh.

---

## ⚖️ Tuyên bố miễn trừ

Đây là dự án học thuật. Mô hình chỉ mang tính **hỗ trợ sàng lọc**, **KHÔNG** thay thế chẩn
đoán của bác sĩ da liễu.

**Nguồn dữ liệu:** Tschandl, P. et al. *The HAM10000 dataset* (ISIC Archive).
