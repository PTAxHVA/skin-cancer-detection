"""Metrics + biểu đồ (confusion matrix, classification report, so sánh model)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chạy headless trong script
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config


def summarize(y_true, y_pred) -> dict:
    """Trả về dict gọn các chỉ số chính (accuracy, macro-F1)."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted")), 4
        ),
    }


def text_report(y_true, y_pred, labels=None, target_names=None) -> str:
    """Classification report dạng text (precision / recall / f1-score)."""
    return classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, digits=3,
        zero_division=0,
    )


def plot_confusion(
    y_true, y_pred, class_names, title: str, filename: str, normalize: bool = False
) -> Path:
    """Vẽ confusion matrix heatmap và lưu ra reports/figures/."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = ".2f"

    fig_w = max(6, len(class_names) * 0.9)
    plt.figure(figsize=(fig_w, fig_w * 0.8))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, cbar=True,
    )
    plt.xlabel("Du doan (Predicted)")
    plt.ylabel("Thuc te (Actual)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = config.FIGURES_DIR / filename
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_class_distribution(counts, title: str, filename: str) -> Path:
    """Bar chart phân bố lớp."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    order = counts.sort_values(ascending=False)
    sns.barplot(x=order.index, y=order.values, palette="viridis")
    for i, v in enumerate(order.values):
        plt.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.ylabel("So luong mau")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = config.FIGURES_DIR / filename
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_model_comparison(results: dict, title: str, filename: str) -> Path:
    """So sánh accuracy & macro-F1 giữa các mô hình (LogReg/KNN/RF/XGB/SVM)."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["f1_macro"] for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, acc, width=0.4, label="Accuracy")
    plt.bar(x + 0.2, f1, width=0.4, label="Macro-F1")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out = config.FIGURES_DIR / filename
    plt.savefig(out, dpi=120)
    plt.close()
    return out
