"""Train SVM (binary + multi-class), lưu model/figures/metrics.

    python -m src.train
"""
from __future__ import annotations

import json
import warnings

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config, evaluation
from .data_preprocessing import add_labels, fit_preprocessor, load_raw

warnings.filterwarnings("ignore")


def _split(df, target):
    return train_test_split(
        df, df[target], test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=df[target],
    )


def _candidates(class_weight):
    rs = config.RANDOM_STATE
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight=class_weight, random_state=rs
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight=class_weight, random_state=rs, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=rs, verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf", class_weight=class_weight, probability=True, random_state=rs
        ),
    }


def _tune_svm(x_train, y_train, class_weight):
    grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.1, 0.01]}
    search = GridSearchCV(
        SVC(kernel="rbf", class_weight=class_weight, probability=True,
            random_state=config.RANDOM_STATE),
        grid, scoring="f1_macro", cv=5, n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search


def _compare(models, x_train, y_train, x_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = evaluation.summarize(y_test, model.predict(x_test))
    return results


def run() -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = add_labels(load_raw())

    evaluation.plot_class_distribution(
        df[config.TARGET_RAW].value_counts(),
        "Phan bo 7 loai ton thuong da (sau khi loai 'healthy')",
        "class_distribution.png",
    )

    metrics: dict = {}

    # Binary
    tr, te, ytr, yte = _split(df, config.TARGET_BINARY)
    x_tr, pre = fit_preprocessor(tr)
    x_te = pre.transform(te)
    ytr, yte = ytr.values, yte.values

    cmp_bin = _compare(_candidates("balanced"), x_tr, ytr, x_te, yte)
    evaluation.plot_model_comparison(
        cmp_bin, "So sanh mo hinh - Binary (class_weight=balanced)",
        "model_comparison_binary.png",
    )

    svm_bin = _tune_svm(x_tr, ytr, "balanced")
    pred_bin = svm_bin.predict(x_te)
    bin_names = [config.BINARY_LABELS[0], config.BINARY_LABELS[1]]
    evaluation.plot_confusion(
        yte, pred_bin, bin_names,
        "Confusion Matrix - SVM Binary", "cm_binary.png",
    )
    metrics["binary"] = {
        "best_params": svm_bin.best_params_,
        "cv_f1_macro": round(float(svm_bin.best_score_), 4),
        "test": evaluation.summarize(yte, pred_bin),
        "model_comparison": cmp_bin,
        "report": evaluation.text_report(yte, pred_bin, target_names=bin_names),
    }

    # Multi-class
    label_enc = LabelEncoder()
    df[config.TARGET_MULTI] = label_enc.fit_transform(df[config.TARGET_RAW])
    class_names = [config.DX_FULL_NAME[c].split(" (")[0] for c in label_enc.classes_]

    tr, te, ytr, yte = _split(df, config.TARGET_MULTI)
    x_tr, pre_m = fit_preprocessor(tr)
    x_te = pre_m.transform(te)
    ytr, yte = ytr.values, yte.values

    cmp_multi = _compare(_candidates("balanced"), x_tr, ytr, x_te, yte)
    evaluation.plot_model_comparison(
        cmp_multi, "So sanh mo hinh - Multi-class (balanced)",
        "model_comparison_multi.png",
    )

    svm_multi = _tune_svm(x_tr, ytr, "balanced")
    pred_multi = svm_multi.predict(x_te)
    evaluation.plot_confusion(
        yte, pred_multi, class_names,
        "Confusion Matrix - SVM Multi-class", "cm_multi.png",
    )
    evaluation.plot_confusion(
        yte, pred_multi, class_names,
        "Confusion Matrix chuan hoa - SVM Multi-class", "cm_multi_norm.png",
        normalize=True,
    )
    metrics["multiclass"] = {
        "best_params": svm_multi.best_params_,
        "cv_f1_macro": round(float(svm_multi.best_score_), 4),
        "test": evaluation.summarize(yte, pred_multi),
        "model_comparison": cmp_multi,
        "report": evaluation.text_report(
            yte, pred_multi, labels=list(range(len(class_names))),
            target_names=class_names,
        ),
    }

    # Save
    joblib.dump(svm_bin.best_estimator_, config.MODELS_DIR / "svm_binary.joblib")
    joblib.dump(svm_multi.best_estimator_, config.MODELS_DIR / "svm_multiclass.joblib")
    joblib.dump(pre, config.MODELS_DIR / "preprocessor_binary.joblib")
    joblib.dump(pre_m, config.MODELS_DIR / "preprocessor_multi.joblib")
    joblib.dump(label_enc, config.MODELS_DIR / "label_encoder_multi.joblib")

    with open(config.MODELS_DIR.parent / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


if __name__ == "__main__":
    m = run()
    print("=== BINARY ===")
    print("best_params:", m["binary"]["best_params"], "| test:", m["binary"]["test"])
    print(m["binary"]["report"])
    print("=== MULTI-CLASS ===")
    print("best_params:", m["multiclass"]["best_params"], "| test:", m["multiclass"]["test"])
    print(m["multiclass"]["report"])
