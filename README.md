---
title: Skin Cancer Detection
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
---

<!-- HF Spaces bỏ SDK streamlit built-in nên chạy Streamlit qua Docker (xem Dockerfile). -->

# Skin Cancer Detection

Phân loại tổn thương da từ metadata bệnh nhân (HAM10000, ISIC) bằng SVM: tiền xử lý →
huấn luyện → đánh giá → cross-validation & tuning, kèm demo Streamlit.

Capstone khoá Machine Learning & Deep Learning — Cybersoft.

## Bài toán

| Bài toán | Mô tả | Nhãn |
|----------|-------|------|
| Binary | Lành tính vs Cần lưu ý | `0` = nv, df, bkl · `1` = mel, bcc, akiec, vasc |
| Multi-class | 7 loại tổn thương | nv, mel, bkl, bcc, akiec, vasc, df |

Dữ liệu: `HAM10000_metadata.csv` (~10k mẫu). Feature: `age`, `sex`, `localization`.

Bỏ `dx_type` và `dataset` (tương quan với nhãn nhưng không có lúc inference → leakage);
bỏ lớp `healthy` (không thuộc 7 loại chuẩn).

## Kết quả

SVM (RBF) + StandardScaler, tuned bằng GridSearchCV (5-fold), `class_weight='balanced'`.

| Bài toán | Best params | CV macro-F1 | Test acc | Test macro-F1 |
|----------|-------------|-------------|----------|---------------|
| Binary | `C=10, gamma=0.01` | 0.651 | 0.714 | 0.633 |
| Multi-class | `C=10, gamma='scale'` | 0.195 | 0.323 | 0.182 |

- Binary đạt recall ~0.58 ở nhóm cần lưu ý (baseline đoán toàn lành tính = 0).
- Multi-class macro-F1 thấp: `age/sex/localization` không đủ tách 7 lớp → cần đặc trưng ảnh.
- Trên macro-F1, SVM ngang RandomForest và hơn XGBoost/KNN (vốn thiên về lớp đa số `nv` ~66%).

Biểu đồ ở `reports/figures/`.

## Cấu trúc

```
skin-cancer-detection/
├── app.py                          # Streamlit demo
├── requirements.txt
├── Dockerfile
├── data/HAM10000_metadata.csv
├── src/
│   ├── config.py                   # đường dẫn, feature, mapping nhãn
│   ├── data_preprocessing.py       # impute, encode, scale
│   ├── train.py                    # train + tuning + lưu model
│   ├── evaluation.py               # metrics, confusion matrix, biểu đồ
│   └── predict.py                  # inference cho app
├── notebooks/skin_cancer_detection.ipynb
├── models/                         # *.joblib
└── reports/                        # metrics.json, figures/
```

## Chạy local

```bash
pip install -r requirements.txt
python -m src.train      # (tuỳ chọn) train lại, sinh models/ + reports/
streamlit run app.py     # http://localhost:8501
```

## Công nghệ

pandas, numpy, scikit-learn (SVC, SimpleImputer, LabelEncoder, StandardScaler,
train_test_split, GridSearchCV), xgboost, matplotlib, seaborn, streamlit, joblib.

## Deploy

Xem [`DEPLOY.md`](DEPLOY.md) (GitHub + Hugging Face Spaces + Streamlit Cloud).

## Hướng mở rộng

1. CNN + transfer learning (ResNet/EfficientNet) trên ảnh thô.
2. API (FastAPI/Flask) + web UI.
3. Multimodal: metadata + đặc trưng ảnh.

## Ghi chú

Dự án học thuật, không thay thế chẩn đoán y khoa.
Dữ liệu: Tschandl, P. et al. *The HAM10000 dataset* (ISIC Archive).
