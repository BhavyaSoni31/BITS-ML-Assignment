"""Train, tune and persist the five classifiers for the Shopper Intent project.

Run from the project root:
    python -m model.train_models

Outputs
    model/<slug>.joblib        fitted pipelines (feature crafting + encoding + classifier)
    model/leaderboard.csv      hold-out metrics for every model
    model/training_report.json tuning details, dataset facts and feature schema
    test_data.csv              untouched hold-out split (features + target) for the app
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.shopper_toolkit import (  # noqa: E402
    CLASS_NAMES,
    METRIC_ORDER,
    RAW_FEATURES,
    TARGET_COLUMN,
    build_feature_pipeline,
    purchase_probability,
    scorecard,
    to_binary_target,
)

RAW_CSV = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
ARTIFACT_DIR = PROJECT_ROOT / "model"
HOLDOUT_CSV = PROJECT_ROOT / "test_data.csv"

SPLIT_SEED = 2611
HOLDOUT_SHARE = 0.20
CV_FOLDS = 5

MODEL_CATALOGUE = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=4000, solver="lbfgs"),
        {"clf__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0], "clf__class_weight": [None, "balanced"]},
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(random_state=SPLIT_SEED),
        {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 5, 7, 10, None],
            "clf__min_samples_leaf": [1, 10, 30, 60],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "knn": (
        "KNN",
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [25, 45, 75, 111, 151],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    ),
    "naive_bayes": (
        "Naive Bayes",
        GaussianNB(),
        {"clf__var_smoothing": [float(v) for v in np.logspace(-9, 0, 10)] + [0.3, 3.0]},
    ),
    "svm": (
        "Support Vector Machine",
        SVC(kernel="rbf", random_state=SPLIT_SEED),
        {
            "clf__C": [0.3, 1.0, 3.0, 10.0],
            "clf__gamma": ["scale", 0.003, 0.01],
            "clf__class_weight": [None, "balanced"],
        },
    ),
}


def load_sessions() -> pd.DataFrame:
    sessions = pd.read_csv(RAW_CSV)
    before = len(sessions)
    sessions = sessions.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {before} rows, removed {before - len(sessions)} exact duplicates -> {len(sessions)} rows")
    return sessions


def tune_one(slug: str, estimator, grid: dict, X_train, y_train, folds) -> tuple[Pipeline, dict]:
    blueprint = build_feature_pipeline()
    blueprint.steps.append(("clf", estimator))
    search = GridSearchCV(
        blueprint,
        param_grid=grid,
        scoring=make_scorer(matthews_corrcoef),
        cv=folds,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    champion = search.best_estimator_

    # SVC has no native probabilities; a sigmoid calibrator fitted on out-of-fold decision
    # scores supplies them (monotonic, so ranking/AUC reflects the SVM itself).
    if slug == "svm":
        tuned_svc = clone(search.best_estimator_.named_steps["clf"])
        champion = clone(search.best_estimator_)
        champion.steps[-1] = ("clf", CalibratedClassifierCV(tuned_svc, method="sigmoid", cv=folds, ensemble=False))
        champion.fit(X_train, y_train)

    tidy_params = {k.replace("clf__", ""): getattr(v, "item", lambda: v)() for k, v in search.best_params_.items()}
    return champion, {"best_params": tidy_params, "cv_mcc": float(search.best_score_)}


def main() -> None:
    sessions = load_sessions()
    features = sessions[RAW_FEATURES]
    target = to_binary_target(sessions[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=HOLDOUT_SHARE, stratify=target, random_state=SPLIT_SEED
    )
    holdout = X_test.copy()
    holdout[TARGET_COLUMN] = y_test.astype(bool)
    holdout.to_csv(HOLDOUT_CSV, index=False)
    print(f"Train: {len(X_train)} rows | Hold-out saved to {HOLDOUT_CSV.name}: {len(X_test)} rows")

    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    leaderboard_rows, tuning_log = [], {}

    for slug, (display_name, estimator, grid) in MODEL_CATALOGUE.items():
        started = time.perf_counter()
        fitted, tuning = tune_one(slug, estimator, grid, X_train, y_train, folds)
        predictions = fitted.predict(X_test)
        scores = scorecard(y_test, predictions, purchase_probability(fitted, X_test))
        elapsed = time.perf_counter() - started

        joblib.dump(fitted, ARTIFACT_DIR / f"{slug}.joblib", compress=3)
        leaderboard_rows.append({"ML Model Name": display_name, **{m: round(scores[m], 4) for m in METRIC_ORDER}})
        tuning_log[slug] = {"display_name": display_name, **tuning, "fit_seconds": round(elapsed, 1)}
        print(f"  {display_name:<24} MCC={scores['MCC']:.4f} AUC={scores['AUC']:.4f} "
              f"(cv MCC {tuning['cv_mcc']:.4f}, {elapsed:.1f}s) params={tuning['best_params']}")

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard.to_csv(ARTIFACT_DIR / "leaderboard.csv", index=False)

    report = {
        "dataset": "UCI Online Shoppers Purchasing Intention (id 468)",
        "rows_after_dedup": int(len(sessions)),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "positive_rate": round(float(target.mean()), 4),
        "split_seed": SPLIT_SEED,
        "cv_folds": CV_FOLDS,
        "selection_metric": "MCC",
        "class_names": list(CLASS_NAMES),
        "raw_features": RAW_FEATURES,
        "target": TARGET_COLUMN,
        "models": tuning_log,
    }
    (ARTIFACT_DIR / "training_report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\nHold-out leaderboard\n" + leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
