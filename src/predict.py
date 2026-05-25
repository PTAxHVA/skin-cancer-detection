"""Load model đã lưu, dự đoán từ age/sex/localization. Dùng bởi app.py."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from . import config


@lru_cache(maxsize=1)
def load_artifacts() -> dict:
    """Load 1 lần và cache lại toàn bộ artifact (model + preprocessor + encoder)."""
    required = [
        "svm_binary.joblib", "svm_multiclass.joblib",
        "preprocessor_binary.joblib", "preprocessor_multi.joblib",
        "label_encoder_multi.joblib",
    ]
    missing = [f for f in required if not (config.MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Thieu file model: " + ", ".join(missing)
            + ". Hay chay 'python -m src.train' truoc."
        )
    return {
        "svm_binary": joblib.load(config.MODELS_DIR / "svm_binary.joblib"),
        "svm_multi": joblib.load(config.MODELS_DIR / "svm_multiclass.joblib"),
        "pre_binary": joblib.load(config.MODELS_DIR / "preprocessor_binary.joblib"),
        "pre_multi": joblib.load(config.MODELS_DIR / "preprocessor_multi.joblib"),
        "label_enc": joblib.load(config.MODELS_DIR / "label_encoder_multi.joblib"),
    }


def _to_frame(age: float, sex: str, localization: str) -> pd.DataFrame:
    return pd.DataFrame([{"age": age, "sex": sex, "localization": localization}])


def predict_binary(age: float, sex: str, localization: str) -> dict:
    """Dự đoán nhị phân + xác suất nhóm 'cần lưu ý'."""
    art = load_artifacts()
    x = art["pre_binary"].transform(_to_frame(age, sex, localization))
    model = art["svm_binary"]
    proba = model.predict_proba(x)[0]
    # Lấy nhãn theo argmax xác suất để label luôn khớp với xác suất hiển thị
    # (SVC.predict dùng decision_function nên có thể lệch predict_proba gần biên).
    label = int(proba.argmax())
    return {
        "label": label,
        "label_name": config.BINARY_LABELS[label],
        "prob_benign": round(float(proba[0]), 4),
        "prob_malignant": round(float(proba[1]), 4),
    }


def predict_multiclass(age: float, sex: str, localization: str, top_k: int = 3) -> dict:
    """Dự đoán loại tổn thương + top-k xác suất cao nhất."""
    art = load_artifacts()
    x = art["pre_multi"].transform(_to_frame(age, sex, localization))
    model, enc = art["svm_multi"], art["label_enc"]
    proba = model.predict_proba(x)[0]
    pred_idx = int(proba.argmax())
    code = enc.inverse_transform([pred_idx])[0]

    ranked = sorted(
        ((enc.inverse_transform([i])[0], float(p)) for i, p in enumerate(proba)),
        key=lambda kv: kv[1], reverse=True,
    )
    return {
        "code": code,
        "full_name": config.DX_FULL_NAME.get(code, code),
        "top_k": [
            {"code": c, "name": config.DX_FULL_NAME.get(c, c), "prob": round(p, 4)}
            for c, p in ranked[:top_k]
        ],
        "all_probs": {c: round(p, 4) for c, p in ranked},
    }
