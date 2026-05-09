from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier(cfg: Dict[str, Any]) -> Pipeline:
    steps = []

    if cfg["model"].get("standardize", True):
        steps.append(("scaler", StandardScaler()))

    pca_n_components = cfg["model"].get("pca_n_components", None)
    if pca_n_components is not None:
        steps.append(("pca", PCA(n_components=pca_n_components)))

    clf = LogisticRegression(
        C=cfg["model"].get("C", 1.0),
        max_iter=cfg["model"].get("max_iter", 2000),
        class_weight=cfg["model"].get("class_weight", None),
        solver=cfg["model"].get("solver", "lbfgs"),
        random_state=cfg.get("seed", 42),
    )

    steps.append(("clf", clf))
    return Pipeline(steps=steps)


def fit_classifier(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    model.fit(X, y)
    return model


def predict_classifier(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def predict_proba_classifier(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba.")
    return model.predict_proba(X)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)