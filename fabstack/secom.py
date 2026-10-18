"""Leakage-aware yield baseline for the attributed UCI SECOM archive.

The label means an observed in-house test failure, not a diagnosed defect cause.
All feature selection and imputation are fitted inside each training fold.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
SHA256 = "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"
ARCHIVE = Path("data/raw/secom.zip")
RESULT = Path("evidence/secom_baseline.json")
SEED = 7
TARGET_PRECISION = 0.5


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Keep columns with at most 50% missingness on the fit subset only."""

    def __init__(self, max_fraction: float = 0.5):
        self.max_fraction = max_fraction

    def fit(self, x, y=None):
        if not 0 <= self.max_fraction < 1:
            raise ValueError("max_fraction must be in [0, 1)")
        self.keep_ = np.isnan(np.asarray(x, dtype=float)).mean(axis=0) <= self.max_fraction
        if not self.keep_.any():
            raise ValueError("No sensor survived the missingness rule")
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[1] != len(self.keep_):
            raise ValueError("Sensor count changed between fit and transform")
        return x[:, self.keep_]


def ensure_archive(path: Path = ARCHIVE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError("UCI SECOM archive hash mismatch")
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("UCI SECOM archive hash mismatch")
    return path


def load_archive(path: Path = ARCHIVE):
    with zipfile.ZipFile(ensure_archive(path)) as archive:
        x = pd.read_csv(io.BytesIO(archive.read("secom.data")), sep=r"\s+", header=None)
        label_lines = archive.read("secom_labels.data").decode().splitlines()
    labels, timestamps = [], []
    for line in label_lines:
        label, timestamp = line.split(maxsplit=1)
        labels.append(int(label))
        timestamps.append(timestamp.strip('"'))
    if x.shape != (1567, 590) or len(labels) != len(x) or set(labels) != {-1, 1}:
        raise ValueError("Unexpected UCI SECOM shape or label encoding")
    when = pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M:%S", errors="raise")
    # The timestamp belongs to the in-house test point. It is only used to order
    # the split; it is never offered to a model as a predictor.
    order = np.argsort(when.to_numpy(), kind="stable")
    return x.to_numpy(dtype=float)[order], (np.asarray(labels)[order] == 1).astype(int), when[order]


def preprocessor() -> Pipeline:
    return Pipeline([
        ("missing", MissingnessFilter(0.5)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("variance", VarianceThreshold(1e-12)),
    ])


def candidates() -> dict[str, Pipeline]:
    from xgboost import XGBClassifier

    return {
        "logistic": Pipeline([
            ("pre", preprocessor()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500)),
        ]),
        "random_forest": Pipeline([
            ("pre", preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=160, min_samples_leaf=3, class_weight="balanced_subsample",
                random_state=SEED, n_jobs=1,
            )),
        ]),
        "xgboost": Pipeline([
            ("pre", preprocessor()),
            ("model", XGBClassifier(
                n_estimators=160, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
                eval_metric="logloss", random_state=SEED, n_jobs=1,
            )),
        ]),
    }


def fit_model(model, x, y, name):
    kwargs = {}
    if name == "xgboost":
        kwargs["model__sample_weight"] = compute_sample_weight("balanced", y)
    return model.fit(x, y, **kwargs)


def threshold_for_precision(y, scores, minimum=TARGET_PRECISION):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(precision[:-1] >= minimum)
    if not len(eligible):
        return 1.0
    best = eligible[np.argmax(recall[eligible])]
    return float(thresholds[best])


def transformed_names(pre: Pipeline) -> list[str]:
    kept = np.flatnonzero(pre.named_steps["missing"].keep_)
    names = [f"sensor_{j:03d}" for j in kept]
    names.extend(f"sensor_{kept[j]:03d}_missing" for j in pre.named_steps["impute"].indicator_.features_)
    return list(np.asarray(names)[pre.named_steps["variance"].get_support()])


def evaluate(path: Path = ARCHIVE) -> dict:
    x, y, when = load_archive(path)
    cut = int(len(x) * 0.8)
    x_train, x_test = x[:cut], x[cut:]
    y_train, y_test = y[:cut], y[cut:]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {}
    for name, prototype in candidates().items():
        fold_metrics = []
        out_of_fold = np.empty(len(y_train), dtype=float)
        for train_idx, valid_idx in folds.split(x_train, y_train):
            model = fit_model(clone(prototype), x_train[train_idx], y_train[train_idx], name)
            scores = model.predict_proba(x_train[valid_idx])[:, 1]
            out_of_fold[valid_idx] = scores
            fold_metrics.append({
                "average_precision": float(average_precision_score(y_train[valid_idx], scores)),
                "roc_auc": float(roc_auc_score(y_train[valid_idx], scores)),
            })
        threshold = threshold_for_precision(y_train, out_of_fold)
        model = fit_model(clone(prototype), x_train, y_train, name)
        scores = model.predict_proba(x_test)[:, 1]
        predicted = scores >= threshold
        top_count = max(1, int(np.ceil(len(y_test) * 0.05)))
        top_indices = np.argsort(-scores, kind="stable")[:top_count]
        top_hits = int(y_test[top_indices].sum())
        models[name] = {
            "cv_folds": fold_metrics,
            "cv_average_precision_mean": float(np.mean([m["average_precision"] for m in fold_metrics])),
            "cv_average_precision_std": float(np.std([m["average_precision"] for m in fold_metrics], ddof=1)),
            "cv_roc_auc_mean": float(np.mean([m["roc_auc"] for m in fold_metrics])),
            "cv_roc_auc_std": float(np.std([m["roc_auc"] for m in fold_metrics], ddof=1)),
            "development_threshold_for_precision_0_5": threshold,
            "test_average_precision": float(average_precision_score(y_test, scores)),
            "test_roc_auc": float(roc_auc_score(y_test, scores)),
            "test_precision_at_development_threshold": float(precision_score(y_test, predicted, zero_division=0)),
            "test_recall_at_development_threshold": float(recall_score(y_test, predicted, zero_division=0)),
            "test_confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_test, predicted, labels=[0, 1]).ravel().tolist(),
            "test_top_5_percent_review": {
                "reviewed": top_count,
                "fails_found": top_hits,
                "precision": top_hits / top_count,
                "recall": top_hits / int(y_test.sum()),
            },
        }
        if name == "xgboost":
            from xgboost import DMatrix

            transformed = model.named_steps["pre"].transform(x_test)
            contributions = model.named_steps["model"].get_booster().predict(
                DMatrix(transformed), pred_contribs=True
            )[:, :-1]
            importance = np.abs(contributions).mean(axis=0)
            names = transformed_names(model.named_steps["pre"])
            models[name]["top_20_tree_shap_log_odds"] = [
                {"feature": names[i], "mean_abs_log_odds": float(importance[i])}
                for i in np.argsort(-importance)[:20]
            ]
    selected = max(models, key=lambda name: models[name]["cv_average_precision_mean"])
    return {
        "source": URL,
        "archive_sha256": SHA256,
        "license": "CC BY 4.0",
        "download_date_sgt": date.today().isoformat(),
        "label": "1 = in-house test fail; -1 = pass; timestamps excluded from predictors",
        "rows": len(x), "raw_sensor_columns": x.shape[1], "fails": int(y.sum()),
        "split": "timestamp order; first 80% development, last 20% untouched test",
        "development_rows": len(y_train), "development_fails": int(y_train.sum()),
        "test_rows": len(y_test), "test_fails": int(y_test.sum()),
        "development_last_timestamp": str(when[cut - 1]),
        "test_first_timestamp": str(when[cut]),
        "selection": "fixed model settings; 5 stratified development folds; threshold chosen on out-of-fold development scores only",
        "selected_by_development_cv_average_precision": selected,
        "baseline_test_average_precision": float(y_test.mean()),
        "models": models,
        "limits": [
            "This is one historical UCI dataset; anonymous sensors cannot be assigned physical root causes.",
            "The held-out tail is chronological by recorded test timestamp, not a prospective deployment or site holdout.",
            "The 0.5 precision threshold is a development target, not a guaranteed test precision or production operating point.",
            "TreeSHAP ranks model contribution magnitude, not causal drivers of yield.",
        ],
    }


def main():
    result = evaluate()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "rows": result["rows"], "fails": result["fails"],
        "test_rows": result["test_rows"], "test_fails": result["test_fails"],
        "model_test_average_precision": {
            name: metrics["test_average_precision"] for name, metrics in result["models"].items()
        },
    }))


if __name__ == "__main__":
    main()
