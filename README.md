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

# Skin Cancer Detection

Phân loại tổn thương da từ metadata HAM10000 (ISIC) bằng SVM, kèm demo Streamlit.
Capstone khoá Machine Learning & Deep Learning — Cybersoft.

## Bài toán

- **Binary**: lành tính (`nv, df, bkl`) vs cần lưu ý (`mel, bcc, akiec, vasc`).
- **Multi-class**: 7 loại — `nv, mel, bkl, bcc, akiec, vasc, df`.

Feature dùng: `age`, `sex`, `localization`. Bỏ `dx_type`/`dataset` (leakage, không có
lúc inference) và lớp `healthy` (không thuộc 7 loại chuẩn).

## Kết quả

SVM (RBF) + StandardScaler, tuned bằng GridSearchCV (5-fold), `class_weight='balanced'`.

| Bài toán | Best params | CV macro-F1 | Test acc | Test macro-F1 |
|----------|-------------|-------------|----------|---------------|
| Binary | `C=10, gamma=0.01` | 0.651 | 0.714 | 0.633 |
| Multi-class | `C=10, gamma='scale'` | 0.195 | 0.323 | 0.182 |

Binary recall ~0.58 ở nhóm cần lưu ý. Multi-class macro-F1 thấp: chỉ với metadata thì
không tách được 7 lớp. Biểu đồ ở `reports/figures/`.

## Chạy

```bash
pip install -r requirements.txt
python -m src.train      # train lại (sinh models/ + reports/)
streamlit run app.py     # http://localhost:8501
```

## Cấu trúc

```
src/        config, data_preprocessing, train, evaluation, predict
notebooks/  skin_cancer_detection.ipynb   (EDA + train + eval)
models/     *.joblib
reports/    metrics.json, figures/
app.py      Streamlit demo
```

Dữ liệu: Tschandl, P. et al. *The HAM10000 dataset* (ISIC). Dự án học thuật, không
thay thế chẩn đoán y khoa.
