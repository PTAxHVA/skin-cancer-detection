# 🎬 Kịch bản video demo — ĐỌC THEO TỪNG CHỮ (~6 phút)

> Cách dùng: bật ghi màn hình rồi **đọc to phần 🎙️** đúng theo thứ tự. Trước mỗi đoạn,
> làm thao tác ở phần **🖥️**. Không cần học thuộc — đọc thẳng là xong.

**2 tab trình duyệt mở sẵn trước khi quay:**
- **Tab A — GitHub:** https://github.com/PTAxHVA/skin-cancer-detection (mở sẵn file `notebooks/skin_cancer_detection.ipynb`)
- **Tab B — Demo:** https://huggingface.co/spaces/PTAxHVA1412/Skin-Cancer-Detection

> 💡 Bấm vào file `.ipynb` trên GitHub là nó tự render kèm biểu đồ — khỏi cần chạy Jupyter.

---

## ⏱️ 0:00 – 0:30 · Mở đầu
🖥️ Hiện **Tab A** (trang GitHub repo, thấy README).

🎙️ *"Em chào thầy/cô. Em là [TÊN], mã học viên [MÃ]. Đây là bài Capstone của em — Phân loại tổn thương da, Skin Cancer Detection, dùng bộ dữ liệu HAM10000 của ISIC. Trong video này em sẽ trình bày nhanh ba phần: bài toán và dữ liệu, quy trình Machine Learning, và demo ứng dụng đã deploy."*

---

## ⏱️ 0:30 – 1:30 · Bài toán & dữ liệu
🖥️ Cuộn README xuống bảng "Bài toán", rồi mở file notebook, cuộn tới biểu đồ phân bố `dx`.

🎙️ *"Em giải quyết hai bài toán. Thứ nhất là phân loại nhị phân: lành tính so với nhóm cần lưu ý. Thứ hai là phân loại bảy loại tổn thương da: nv, mel, bkl, bcc, akiec, vasc và df.*

*Điểm quan trọng: đề yêu cầu dùng đặc trưng trích xuất sẵn từ file metadata, chứ không dùng ảnh thô. Dữ liệu có khoảng mười nghìn mẫu.*

*Nhìn vào biểu đồ phân bố này, thầy/cô có thể thấy lớp nv — nốt ruồi lành — chiếm tới sáu mươi sáu phần trăm. Tức là dữ liệu mất cân bằng rất nặng. Đây là điều em phải xử lý cẩn thận ở các bước sau."*

---

## ⏱️ 1:30 – 3:00 · Quy trình ML — phần trọng tâm ⭐
🖥️ Cuộn notebook tới mục "3.4 — heatmap dx_type", rồi tới mục tiền xử lý, rồi mục so sánh SVM mặc định và balanced.

🎙️ *"Trước khi huấn luyện, em phân tích để chọn đặc trưng. Cột dx_type — mô tả cách bác sĩ xác nhận chẩn đoán — có tương quan rất mạnh với nhãn, như heatmap này cho thấy. Nếu đưa vào mô hình thì độ chính xác sẽ tăng ảo, đây gọi là rò rỉ dữ liệu, data leakage. Và lúc người dùng thực tế thì cũng không có thông tin này. Nên em loại bỏ dx_type và dataset, chỉ giữ ba đặc trưng phía bệnh nhân: tuổi, giới tính, và vị trí tổn thương.*

*Các bước tiền xử lý gồm: điền giá trị thiếu cho cột tuổi bằng SimpleImputer, mã hóa biến phân loại bằng LabelEncoder, và chuẩn hóa bằng StandardScaler.*

*Một điểm em muốn nhấn mạnh là accuracy gây hiểu lầm với dữ liệu mất cân bằng. SVM mặc định cho accuracy cao hơn nhưng gần như chỉ đoán toàn lành tính, bỏ sót ca bệnh. Khi em thêm class_weight balanced, accuracy giảm một chút nhưng mô hình phát hiện được lớp ác tính. Trong y khoa, thà báo nhầm còn hơn bỏ sót, nên em ưu tiên hướng này."*

---

## ⏱️ 3:00 – 4:15 · Kết quả & đánh giá
🖥️ Cuộn tới confusion matrix + classification report của binary, rồi multi-class, rồi bảng so sánh mô hình và GridSearchCV.

🎙️ *"Với bài toán nhị phân, sau khi tinh chỉnh siêu tham số bằng GridSearchCV, mô hình SVM đạt độ chính xác khoảng bảy mươi mốt phần trăm và macro F1 khoảng không phẩy sáu ba. Quan trọng hơn, nó phát hiện được khoảng năm mươi tám phần trăm số ca cần lưu ý — trong khi mô hình ngây thơ đoán toàn lành tính thì phát hiện được không phần trăm. Confusion matrix và classification report ở đây thể hiện rõ precision, recall và f1 cho từng lớp.*

*Với bài toán bảy lớp, macro F1 chỉ khoảng không phẩy mười tám. Em xin thẳng thắn: chỉ với metadata nhân khẩu học thì không đủ để phân biệt bảy loại bệnh. Đây chính là lý do em đề xuất hướng mở rộng dùng CNN trên ảnh thô.*

*Cuối cùng, em có so sánh năm mô hình — Logistic Regression, KNN, Random Forest, XGBoost và SVM. Xét trên macro F1 là chỉ số đúng cho dữ liệu mất cân bằng, thì SVM cạnh tranh ngang Random Forest và tốt hơn các mô hình chỉ tối ưu accuracy."*

---

## ⏱️ 4:15 – 5:45 · Demo ứng dụng đã deploy ⭐
🖥️ Chuyển sang **Tab B** (HF Space). Đợi app hiện form.

🎙️ *"Bây giờ em demo ứng dụng đã được triển khai trên Hugging Face Spaces bằng Streamlit."*

🖥️ Nhập **Tuổi = 25**, **Giới tính = female**, **Vị trí = lower extremity** → bấm **Dự đoán**.

🎙️ *"Trường hợp một: một bệnh nhân nữ hai mươi lăm tuổi, tổn thương ở chân dưới. Mô hình dự đoán nghiêng hẳn về lành tính, với xác suất cao. Kết quả này hợp lý."*

🖥️ Đổi **Tuổi = 80**, **Giới tính = male**, **Vị trí = face** → bấm **Dự đoán**.

🎙️ *"Trường hợp hai: một bệnh nhân nam tám mươi tuổi, tổn thương ở vùng mặt. Lần này xác suất 'cần lưu ý' tăng lên rõ rệt và mô hình khuyến nghị thăm khám chuyên khoa. Điều này cũng phù hợp về mặt y khoa, vì tổn thương vùng mặt ở người lớn tuổi có nguy cơ cao hơn."*

🖥️ Cuộn xuống phần top-3 + mở mục "Xem toàn bộ xác suất 7 lớp"; rồi chỉ vào dòng tuyên bố miễn trừ màu vàng.

🎙️ *"Ứng dụng cũng hiển thị ba khả năng cao nhất cho bài toán bảy lớp, và biểu đồ xác suất đầy đủ. Em cũng để rõ tuyên bố miễn trừ: đây chỉ là công cụ hỗ trợ sàng lọc, không thay thế chẩn đoán của bác sĩ."*

---

## ⏱️ 5:45 – 6:30 · Kết & hướng mở rộng
🖥️ Quay lại **Tab A** (GitHub), cuộn tới mục "Hướng mở rộng".

🎙️ *"Tóm lại, em đã xây dựng một pipeline Machine Learning hoàn chỉnh từ tiền xử lý, huấn luyện, tinh chỉnh đến đánh giá, kèm một ứng dụng demo trực tuyến. Hướng mở rộng tiếp theo là dùng CNN với transfer learning trên ảnh thô, và xây dựng API. Toàn bộ mã nguồn ở repository GitHub này, và bản demo ở trên Hugging Face Spaces. Em cảm ơn thầy/cô đã lắng nghe."*

---

## ✅ Checklist quay
- [ ] Đã đăng nhập sẵn, đóng tab thừa, ẩn bookmark/thông báo cá nhân.
- [ ] Phóng to chữ trình duyệt (Ctrl/Cmd +) cho dễ đọc.
- [ ] Quay 1080p; công cụ: QuickTime (macOS) hoặc OBS.
- [ ] Mở app HF Space **trước 1–2 phút** cho nó "thức dậy" (Space free ngủ khi không dùng).
- [ ] Xuất video → upload YouTube để **Public** hoặc **Unlisted** → dán link vào form Nộp bài.

## 📌 Số liệu & link để sẵn (phòng khi cần đọc lại)
- Binary: accuracy ≈ **71%**, macro-F1 ≈ **0.63**, recall ca cần lưu ý ≈ **58%**, best `C=10, gamma=0.01`.
- Multi-class: accuracy ≈ **32%**, macro-F1 ≈ **0.18** (minh chứng cần CNN).
- Dataset: HAM10000, ~**10.000** mẫu, lớp `nv` ≈ **66%**.
- Source: https://github.com/PTAxHVA/skin-cancer-detection
- Demo: https://huggingface.co/spaces/PTAxHVA1412/Skin-Cancer-Detection
