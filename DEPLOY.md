# 🚀 Hướng dẫn nộp bài Capstone — 3 link

Bài nộp cần **3 link**: ① Link nộp source · ② Link deploy · ③ Link YouTube.
Làm theo đúng thứ tự dưới đây.

---

## ① Link nộp source → GitHub

```bash
cd skin-cancer-detection

git init
git add .
git commit -m "feat: capstone skin cancer detection (SVM + Streamlit)"

# Tao repo tren GitHub (vd: ten 'skin-cancer-detection'), roi:
git branch -M main
git remote add origin https://github.com/<TEN_GITHUB>/skin-cancer-detection.git
git push -u origin main
```

➡️ **Link nộp source** = `https://github.com/<TEN_GITHUB>/skin-cancer-detection`

> 💡 Chưa có GitHub CLI? Cài `gh` rồi `gh repo create skin-cancer-detection --public --source=. --push`
> là tạo repo + push một lệnh.

---

## ② Link deploy → Hugging Face Spaces (khuyến nghị, giống demo mẫu)

File `README.md` đã có sẵn **frontmatter** (`sdk: streamlit`, `app_file: app.py`) nên Space
tự nhận cấu hình.

**Cách A — Web (dễ nhất):**
1. Vào https://huggingface.co/new-space → đặt tên `Skin-Cancer-Detection`.
2. Chọn **SDK = Streamlit**, visibility **Public**, bấm *Create Space*.
3. Tab **Files** → *Add file* → *Upload files* → kéo thả **toàn bộ** nội dung thư mục
   (gồm `app.py`, `src/`, `models/`, `data/`, `requirements.txt`, `README.md`).
4. Đợi build (~2–4 phút). Space tự chạy `streamlit run app.py`.

**Cách B — Git:**
```bash
pip install huggingface_hub
huggingface-cli login          # dan token tu https://huggingface.co/settings/tokens
git remote add space https://huggingface.co/spaces/<USER>/Skin-Cancer-Detection
git push space main
```

➡️ **Link deploy** = `https://huggingface.co/spaces/<USER>/Skin-Cancer-Detection`

### Thay thế: Streamlit Community Cloud
1. Push code lên GitHub (bước ①).
2. Vào https://share.streamlit.io → *New app* → chọn repo, branch `main`, file `app.py`.
3. *Deploy*. ➡️ Link dạng `https://<app>.streamlit.app`.

> ⚠️ Bắt buộc commit thư mục `models/` (các file `.joblib`) — demo cần để dự đoán.
> `requirements.txt` đã ghim `scikit-learn>=1.7,<1.8` để model load đúng.

---

## ③ Link YouTube → video demo

1. Quay màn hình theo kịch bản trong **[`YOUTUBE_SCRIPT.md`](YOUTUBE_SCRIPT.md)** (5–8 phút).
2. Upload lên YouTube, để chế độ **Public** hoặc **Unlisted**.
3. ➡️ **Link YouTube** = link video.

---

## ✅ Checklist trước khi bấm "Nộp bài" (deadline 25/05/2026)

- [ ] GitHub repo public, có README + notebook + `src/` + `models/`.
- [ ] HF Space / Streamlit app build xanh, bấm "Dự đoán" ra kết quả.
- [ ] Video YouTube xem được (không để Private).
- [ ] Dán đủ 3 link vào form và *Nộp bài*.
