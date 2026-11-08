"""Reusable pieces shared by the training script and the Streamlit front-end.

Keeping the feature schema, the engineered-feature transformer and the metric
calculation in one place guarantees that the app scores uploaded data exactly
the way the models were evaluated during training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

TARGET_COLUMN = "Revenue"
CLASS_NAMES = ("No Purchase", "Purchase")

PAGE_COUNTS = ["Administrative", "Informational", "ProductRelated"]
PAGE_DURATIONS = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]
GA_SIGNALS = ["BounceRates", "ExitRates", "PageValues", "SpecialDay"]
SESSION_CONTEXT = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]

RAW_FEATURES = PAGE_COUNTS + PAGE_DURATIONS + GA_SIGNALS + SESSION_CONTEXT
CRAFTED_SIGNALS = ["TotalPages", "TotalDuration", "SecondsPerProductPage", "HasPageValue"]
NUMERIC_BLOCK = PAGE_COUNTS + PAGE_DURATIONS + GA_SIGNALS + CRAFTED_SIGNALS

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

_TRUTHY = {"1", "1.0", "true", "t", "yes", "y", "purchase"}
_FALSY = {"0", "0.0", "false", "f", "no", "n", "no purchase"}


def _canonical_code(series: pd.Series) -> pd.Series:
    """Map category codes such as 2, 2.0, '2', True, 'TRUE' to one spelling."""
    if pd.api.types.is_bool_dtype(series):
        return series.map({True: "true", False: "false"})
    text = series.astype(str).str.strip().str.lower()
    as_number = pd.to_numeric(series, errors="coerce")
    numeric_rows = as_number.notna()
    text[numeric_rows] = as_number[numeric_rows].round().astype("int64").astype(str)
    return text


class SessionFeatureCrafter(BaseEstimator, TransformerMixin):
    """Cleans raw session columns and appends behaviour-based features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for col in PAGE_COUNTS + PAGE_DURATIONS + GA_SIGNALS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame["TotalPages"] = frame[PAGE_COUNTS].sum(axis=1)
        frame["TotalDuration"] = frame[PAGE_DURATIONS].sum(axis=1)
        frame["SecondsPerProductPage"] = frame["ProductRelated_Duration"] / (frame["ProductRelated"] + 1.0)
        frame["HasPageValue"] = (frame["PageValues"] > 0).astype(float)

        for col in SESSION_CONTEXT:
            frame[col] = _canonical_code(frame[col])
        return frame[NUMERIC_BLOCK + SESSION_CONTEXT]


def build_feature_pipeline() -> Pipeline:
    """Crafting -> (Yeo-Johnson numeric | one-hot categorical)."""
    numeric_branch = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("reshape", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )
    categorical_branch = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=15, sparse_output=False
    )
    column_router = ColumnTransformer(
        [
            ("numeric", numeric_branch, NUMERIC_BLOCK),
            ("context", categorical_branch, SESSION_CONTEXT),
        ]
    )
    return Pipeline([("craft", SessionFeatureCrafter()), ("encode", column_router)])


def to_binary_target(values: pd.Series) -> pd.Series:
    """Accepts True/False, 1/0 or text labels and returns 0/1 integers."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(int)
    tokens = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(tokens) - _TRUTHY - _FALSY)
    if unknown:
        raise ValueError(f"Unrecognised target values in '{TARGET_COLUMN}': {unknown[:5]}")
    return tokens.isin(_TRUTHY).astype(int)


def purchase_probability(estimator, features: pd.DataFrame) -> np.ndarray:
    """Score for the positive (Purchase) class used for AUC and ROC curves."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.decision_function(features)


def scorecard(y_true, y_pred, y_score) -> dict[str, float]:
    """Six assignment metrics; Precision/Recall/F1 are for the Purchase class."""
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if has_both_classes else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
