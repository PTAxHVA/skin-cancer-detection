"""Streamlit demo - Skin Cancer Detection (phân loại từ metadata bệnh nhân).

Chạy local:   streamlit run app.py
Deploy:       Hugging Face Spaces (SDK=streamlit) hoặc Streamlit Community Cloud.

Mô hình dự đoán từ thông tin bệnh nhân (tuổi, giới tính, vị trí tổn thương),
KHÔNG phải từ ảnh thô -> đây là công cụ HỖ TRỢ SÀNG LỌC, không thay thế chẩn đoán.
"""
from __future__ import annotations

import streamlit as st

from src import config, predict

st.set_page_config(
    page_title="Skin Cancer Detection", page_icon="🔬", layout="centered"
)

LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "leg", "lower extremity", "neck", "scalp", "trunk", "unknown",
    "upper extremity",
]
SEX_OPTIONS = ["male", "female", "unknown"]
BENIGN_VI = "Lành tính (Benign)"
MALIGNANT_VI = "Cần lưu ý (Malignant-like)"


def render_header() -> None:
    st.title("🔬 Skin Cancer Detection")
    st.caption(
        "Capstone ML — phân loại tổn thương da từ metadata bệnh nhân (HAM10000). "
        "SVM + tiền xử lý scikit-learn."
    )
    st.warning(
        "⚠️ **Tuyên bố miễn trừ:** Đây là công cụ HỖ TRỢ SÀNG LỌC dựa trên đặc trưng "
        "nhân khẩu học (tuổi/giới/vị trí), **không thay thế chẩn đoán y khoa**. Hãy đến "
        "bác sĩ da liễu để được khám chính xác.",
        icon="⚠️",
    )


def render_inputs():
    st.subheader("📋 Thông tin bệnh nhân")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Tuổi (age)", 0, 100, 50, step=1)
        sex = st.selectbox("Giới tính (sex)", SEX_OPTIONS, index=0)
    with col2:
        localization = st.selectbox(
            "Vị trí tổn thương (localization)", LOCALIZATIONS,
            index=LOCALIZATIONS.index("back"),
        )
        st.file_uploader(
            "Ảnh tổn thương (tuỳ chọn — chỉ để tham khảo, không dùng dự đoán)",
            type=["jpg", "jpeg", "png"],
        )
    return age, sex, localization


def render_binary(age, sex, localization) -> None:
    res = predict.predict_binary(age, sex, localization)
    st.subheader("1️⃣ Phân loại nhị phân: Lành tính vs Cần lưu ý")
    if res["label"] == 1:
        st.error(
            f"🔴 **{MALIGNANT_VI}** — xác suất {res['prob_malignant']:.1%}. "
            "Nên thăm khám chuyên khoa.",
            icon="🔴",
        )
    else:
        st.success(
            f"🟢 **{BENIGN_VI}** — xác suất lành tính {res['prob_benign']:.1%}.",
            icon="🟢",
        )
    st.progress(
        res["prob_malignant"], text=f"Mức độ 'cần lưu ý': {res['prob_malignant']:.1%}"
    )


def render_multiclass(age, sex, localization) -> None:
    res = predict.predict_multiclass(age, sex, localization, top_k=3)
    st.subheader("2️⃣ Phân loại 7 loại tổn thương da")
    st.markdown(f"**Dự đoán khả năng cao nhất:** `{res['code']}` — {res['full_name']}")
    st.caption("Top 3 khả năng (độ tin cậy thường thấp với metadata-only):")
    for item in res["top_k"]:
        st.progress(
            item["prob"], text=f"{item['code']} — {item['name']}: {item['prob']:.1%}"
        )
    with st.expander("Xem toàn bộ xác suất 7 lớp"):
        st.bar_chart(res["all_probs"])


def render_about() -> None:
    with st.expander("ℹ️ Về mô hình & giới hạn"):
        st.markdown(
            "- **Dữ liệu:** HAM10000_metadata.csv (~10k mẫu, ISIC).\n"
            "- **Feature:** age, sex, localization (đặc trưng phía bệnh nhân).\n"
            "- **Mô hình:** SVM (RBF) + StandardScaler, tune bằng GridSearchCV, "
            "`class_weight='balanced'` để phát hiện lớp hiếm.\n"
            "- **Giới hạn:** metadata nhân khẩu học có tín hiệu yếu → binary dùng được "
            "để sàng lọc, multi-class độ tin cậy thấp. Giải pháp thực sự là "
            "**CNN trên ảnh thô** (hướng mở rộng)."
        )


def main() -> None:
    render_header()
    age, sex, localization = render_inputs()
    if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
        try:
            render_binary(age, sex, localization)
            render_multiclass(age, sex, localization)
        except FileNotFoundError as err:
            st.error(f"Chưa có model: {err}")
    render_about()


if __name__ == "__main__":
    main()
