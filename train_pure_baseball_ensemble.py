"""Train the production pure-baseball tree-blend model.

This model uses no odds, moneylines, betting features, or market-aware inputs.
It trains calibrated CatBoost variants on the same market-independent baseball
feature set, then saves one artifact that blends their predicted home-win
probabilities.

Output:
  models/win_prob_blend_trees.pkl
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import warnings
from datetime import datetime, timezone
from importlib import metadata

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

import database as db
from feature_defaults import apply_feature_defaults
from train_model import MARKET_INDEPENDENT_FEATURE_COLS, make_sample_weights

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(MODEL_DIR, "win_prob_blend_trees.pkl")
MANIFEST_PATH = os.path.join(MODEL_DIR, "win_prob_blend_trees.manifest.json")
REGISTRY_PATH = os.path.join(DATA_DIR, "model_registry.csv")


def calibrated(estimator) -> CalibratedClassifierCV:
    cv = TimeSeriesSplit(n_splits=3)
    try:
        return CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method="isotonic", cv=cv)


def model_specs() -> dict:
    return {
        "cat_current": CatBoostClassifier(
            iterations=250,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        ),
        "cat_reg": CatBoostClassifier(
            iterations=450,
            depth=3,
            learning_rate=0.02,
            l2_leaf_reg=12.0,
            random_strength=1.0,
            bagging_temperature=0.3,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=43,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def blend_weights() -> dict[str, float]:
    return {
        "cat_current": 0.5,
        "cat_reg": 0.5,
    }


def load_training_data() -> pd.DataFrame:
    features = db.load_features()
    features = apply_feature_defaults(features, MARKET_INDEPENDENT_FEATURE_COLS)
    features = features[features["home_win"].notna()].copy()
    features["Date"] = pd.to_datetime(features["Date"])
    return features.sort_values("Date").reset_index(drop=True)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_manifest(artifact: dict, model_path: str) -> dict:
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "model_id": f"pure_baseball_blend_trees_{trained_at.replace(':', '').replace('+', 'Z')}",
        "model_path": model_path,
        "model_sha256": file_sha256(model_path),
        "model_type": artifact["model_type"],
        "blend_method": artifact["blend_method"],
        "base_models": artifact["base_models"],
        "blend_weights": artifact.get("blend_weights"),
        "uses_market": artifact["uses_market"],
        "target": artifact["target"],
        "feature_count": len(artifact["features"]),
        "trained_rows": artifact["trained_rows"],
        "trained_year_min": artifact["trained_year_min"],
        "trained_year_max": artifact["trained_year_max"],
        "trained_date_max": artifact["trained_date_max"],
        "trained_at_utc": trained_at,
        "python": platform.python_version(),
        "packages": {
            "catboost": package_version("catboost"),
            "lightgbm": package_version("lightgbm"),
            "xgboost": package_version("xgboost"),
            "scikit-learn": package_version("scikit-learn"),
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
        },
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    registry_row = pd.DataFrame([{
        "model_id": manifest["model_id"],
        "model_type": manifest["model_type"],
        "model_path": manifest["model_path"],
        "model_sha256": manifest["model_sha256"],
        "trained_at_utc": manifest["trained_at_utc"],
        "trained_rows": manifest["trained_rows"],
        "trained_year_min": manifest["trained_year_min"],
        "trained_year_max": manifest["trained_year_max"],
        "trained_date_max": manifest["trained_date_max"],
        "feature_count": manifest["feature_count"],
        "base_models": ",".join(manifest["base_models"]),
    }])
    if os.path.exists(REGISTRY_PATH):
        registry = pd.read_csv(REGISTRY_PATH)
        registry = pd.concat([registry, registry_row], ignore_index=True)
        registry = registry.drop_duplicates(["model_id"], keep="last")
    else:
        registry = registry_row
    registry.to_csv(REGISTRY_PATH, index=False)
    return manifest


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    train = load_training_data()
    available_features = [c for c in MARKET_INDEPENDENT_FEATURE_COLS if c in train.columns]
    X = train[available_features]
    y = train["home_win"].astype(int)
    weights = make_sample_weights(train["year"])

    fitted = {}
    for name, estimator in model_specs().items():
        print(f"Training {name} on {len(train):,} completed games...")
        model = calibrated(estimator)
        model.fit(X, y, sample_weight=weights)
        fitted[name] = model

    artifact = {
        "model_type": "pure_baseball_blend_trees",
        "blend_method": "weighted_mean_probability",
        "blend_weights": blend_weights(),
        "uses_market": False,
        "target": "home_win",
        "features": available_features,
        "pipelines": fitted,
        "base_models": list(fitted.keys()),
        "trained_rows": int(len(train)),
        "trained_year_min": int(train["year"].min()),
        "trained_year_max": int(train["year"].max()),
        "trained_date_max": str(pd.to_datetime(train["Date"]).max().date()),
    }
    joblib.dump(artifact, OUT_PATH)
    manifest = write_manifest(artifact, OUT_PATH)
    print(f"Saved pure baseball blend model -> {OUT_PATH}")
    print(f"Saved model manifest -> {MANIFEST_PATH}")
    print(f"Registered model id -> {manifest['model_id']}")
    print(
        f"Rows: {artifact['trained_rows']:,} | "
        f"Years: {artifact['trained_year_min']}-{artifact['trained_year_max']} | "
        f"Latest date: {artifact['trained_date_max']}"
    )


if __name__ == "__main__":
    main()
