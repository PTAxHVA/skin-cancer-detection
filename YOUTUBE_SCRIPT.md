# 🎬 Kịch bản video demo (5–8 phút)

Quay màn hình + giọng thuyết minh. Mục tiêu: chứng minh hiểu pipeline ML, không chỉ "chạy được".

---

## 0. Mở đầu (0:00–0:30)
> "Chào thầy/cô, em là [TÊN]. Đây là Capstone phân loại tổn thương da dùng bộ HAM10000.
> Em xây dựng pipeline Machine Learning hoàn chỉnh với SVM và demo bằng Streamlit."

Chiếu nhanh trang GitHub repo + README.

## 1. Bài toán & dữ liệu (0:30–1:30)
- Mở `notebooks/skin_cancer_detection.ipynb`.
- Nói rõ 2 bài toán: **binary** (lành tính/cần lưu ý) và **multi-class** (7 loại).
- Chỉ vào biểu đồ phân bố `dx`: **"nv chiếm 66% → dữ liệu mất cân bằng nặng"**.

## 2. Điểm nhấn — tư duy ML (1:30–3:00)  ⭐ phần ăn điểm
- **Data leakage:** chỉ vào heatmap `dx_type` vs `dx`, giải thích vì sao **loại** `dx_type`
  và `dataset` (rò rỉ + không có lúc dự đoán). Chỉ giữ `age, sex, localization`.
- **Tiền xử lý:** SimpleImputer (age) → LabelEncoder → StandardScaler.
- **Accuracy paradox:** so sánh SVM mặc định (acc cao, macro-F1 thấp) vs SVM `balanced`
  (acc thấp hơn nhưng phát hiện được ca ác tính). "Trong y khoa, thà báo nhầm còn hơn bỏ sót."

## 3. Huấn luyện & đánh giá (3:00–4:30)
- Chạy cell SVM binary → chỉ **confusion matrix** + **classification report** (precision/recall/f1).
- Chạy multi-class → confusion matrix 7 lớp + bản chuẩn hoá. Thừa nhận thẳng thắn:
  **"metadata không đủ tách 7 loại → đây là lý do hướng mở rộng dùng CNN trên ảnh."**
- Chỉ bảng **so sánh 5 mô hình** và **GridSearchCV** (best params, CV score).

## 4. Demo Streamlit (4:30–6:30)  ⭐ phần trực quan
- `streamlit run app.py` (hoặc mở link HF Space đã deploy).
- Nhập 1 ca **trẻ, nốt ở chân** → kết quả thiên "lành tính".
- Nhập 1 ca **già, tổn thương ở mặt** → xác suất "cần lưu ý" tăng. Giải thích hợp lý y khoa.
- Chỉ thanh xác suất + top-3 loại bệnh + biểu đồ 7 lớp.
- Đọc **tuyên bố miễn trừ** (không thay thế bác sĩ).

## 5. Kết & hướng mở rộng (6:30–7:30)
> "Tóm lại em đã hoàn thiện pipeline ML đầy đủ, phân tích trung thực giới hạn của
> metadata, và đề xuất CNN trên ảnh thô làm bước tiếp theo. Em cảm ơn thầy/cô."

Chiếu lại 3 link: GitHub · HF Space · (chính video này).

---

## Mẹo quay
- Công cụ: QuickTime (macOS) / OBS (miễn phí). Đặt độ phân giải ≥ 1080p.
- Phóng to font notebook/terminal cho dễ đọc.
- Chạy thử `python -m src.train` và `streamlit run app.py` **trước** khi quay để mượt.
- Nếu lần đầu chạy notebook chậm, chạy hết 1 lượt trước rồi quay lượt 2.
