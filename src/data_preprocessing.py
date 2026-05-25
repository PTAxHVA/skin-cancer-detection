"""Tiền xử lý HAM10000: impute age, encode categorical, scale."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config


@dataclass
class Preprocessor:
    """Imputer + encoders + scaler đã fit, dùng lại lúc inference."""

    age_imputer: SimpleImputer
    encoders: dict[str, LabelEncoder]
    scaler: StandardScaler
    feature_columns: list[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.feature_columns].copy()
        out["age"] = self.age_imputer.transform(out[["age"]])
        for col, encoder in self.encoders.items():
            # Giá trị lạ -> fallback về class đầu tiên thay vì raise.
            known = set(encoder.classes_)
            out[col] = out[col].astype(str).apply(
                lambda v: v if v in known else encoder.classes_[0]
            )
            out[col] = encoder.transform(out[col])
        return pd.DataFrame(
            self.scaler.transform(out),
            columns=self.feature_columns,
            index=out.index,
        )


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay du lieu: {path}. Dat HAM10000_metadata.csv vao data/."
        )
    return pd.read_csv(path)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Loại 'healthy', tạo nhãn binary + multi."""
    df = df[~df[config.TARGET_RAW].isin(config.DROP_CLASSES)].copy()
    df[config.TARGET_BINARY] = (
        df[config.TARGET_RAW].isin(config.MALIGNANT_GROUP).astype(int)
    )
    df[config.TARGET_MULTI] = df[config.TARGET_RAW]
    return df.reset_index(drop=True)


def fit_preprocessor(df: pd.DataFrame) -> tuple[pd.DataFrame, Preprocessor]:
    """Fit trên train, trả về (X, preprocessor)."""
    features = df[config.FEATURE_COLUMNS].copy()

    age_imputer = SimpleImputer(strategy="median")
    features["age"] = age_imputer.fit_transform(features[["age"]])

    encoders: dict[str, LabelEncoder] = {}
    for col in config.CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(features[col].astype(str))
        encoders[col] = encoder

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    x = pd.DataFrame(scaled, columns=config.FEATURE_COLUMNS, index=features.index)

    pre = Preprocessor(age_imputer, encoders, scaler, config.FEATURE_COLUMNS)
    return x, pre
