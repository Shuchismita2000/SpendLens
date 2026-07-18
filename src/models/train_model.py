"""
train_model.py
==============
Fits SpendLens's core model:
    raw spend --[geometric adstock]--> --[Hill saturation]--> media features
    media features + controls --[ElasticNet]--> units_sold

Adstock/saturation params (lambda, k, s per channel) and the ElasticNet
penalty (alpha, l1_ratio) are searched JOINTLY via randomized search, scored
by expanding-window CV -- this is the "no manual VIF pruning, fit in one
pass" requirement: nothing about the media transforms is hand-tuned before
the regression sees the data.

Adstock/saturation are causal (adstocked_t depends only on spend_1..t), so
they're computed ONCE over the full historical series per hyperparameter
draw -- this is correct, not leakage, because no future information enters
week t's transformed value. Only the ElasticNet fit + feature scaling are
fold-scoped (scaler fit on train fold only, as usual).

Run:
    python src/models/train_model.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config  # noqa: E402
from src.data.preprocess import preprocess  # noqa: E402
from src.features.build_features import build_feature_matrix  # noqa: E402
from src.features.adstock import GeometricAdstock  # noqa: E402
from src.features.saturation import HillSaturation  # noqa: E402
from src.models.cv_utils import expanding_window_splits  # noqa: E402


def sample_hyperparams(rng: np.random.Generator, config: dict, channel_cols: list, max_spend: dict):
    hp = config["hyperparameter_search"]
    lam_lo, lam_hi = hp["adstock_lambda_bounds"]
    s_lo, s_hi = hp["saturation_s_bounds"]
    kf_lo, kf_hi = hp["saturation_k_frac_bounds"]
    a_lo, a_hi = hp["elasticnet_alpha_bounds_log10"]
    l1_lo, l1_hi = hp["elasticnet_l1_ratio_bounds"]

    lambdas = {i: rng.uniform(lam_lo, lam_hi) for i in range(len(channel_cols))}
    s_params = {i: rng.uniform(s_lo, s_hi) for i in range(len(channel_cols))}
    k_params = {}
    for i, col in enumerate(channel_cols):
        k_frac = rng.uniform(kf_lo, kf_hi)
        k_params[i] = max(1.0, k_frac * max_spend[col])

    alpha = 10 ** rng.uniform(a_lo, a_hi)
    l1_ratio = rng.uniform(l1_lo, l1_hi)
    return {"lambdas": lambdas, "s": s_params, "k": k_params, "alpha": alpha, "l1_ratio": l1_ratio}


def transform_media(X_channels: np.ndarray, lambdas: dict, k: dict, s: dict) -> np.ndarray:
    adstocked = GeometricAdstock(lambdas=lambdas).fit_transform(X_channels)
    saturated = HillSaturation(k=k, s=s).fit_transform(adstocked)
    return saturated


def evaluate_hyperparams(
    media_raw: np.ndarray, controls: np.ndarray, y: np.ndarray, params: dict, cv_splits: list
) -> dict:
    media_transformed = transform_media(media_raw, params["lambdas"], params["k"], params["s"])
    X_full = np.hstack([media_transformed, controls])

    fold_r2, fold_mape = [], []
    for train_idx, test_idx in cv_splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_full[train_idx])
        X_test = scaler.transform(X_full[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=5000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_r2.append(r2_score(y_test, preds))
        fold_mape.append(mean_absolute_percentage_error(y_test, preds))

    return {"mean_r2": float(np.mean(fold_r2)), "mean_mape": float(np.mean(fold_mape)),
            "fold_r2": fold_r2, "fold_mape": fold_mape}


def run_search(config: dict) -> dict:
    processed = preprocess(config)
    bundle = build_feature_matrix(processed, config)
    X, y = bundle["X"], bundle["y"].values
    channel_cols = bundle["channel_cols"]
    continuous_cols = config["controls"]["continuous"]
    binary_cols = config["controls"]["binary"]

    media_raw = X[channel_cols].values.astype(float)
    controls = X[continuous_cols + binary_cols].values.astype(float)
    max_spend = {c: X[c].max() for c in channel_cols}

    cv_cfg = config["cross_validation"]
    cv_splits = expanding_window_splits(len(y), cv_cfg["min_train_weeks"], cv_cfg["n_splits"])

    hp_cfg = config["hyperparameter_search"]
    rng = np.random.default_rng(hp_cfg["random_state"])

    best = {"score": -np.inf, "params": None, "metrics": None}
    all_trials = []
    for i in range(hp_cfg["n_iter"]):
        params = sample_hyperparams(rng, config, channel_cols, max_spend)
        metrics = evaluate_hyperparams(media_raw, controls, y, params, cv_splits)
        all_trials.append({"iter": i, "mean_r2": metrics["mean_r2"], "mean_mape": metrics["mean_mape"]})
        if metrics["mean_r2"] > best["score"]:
            best = {"score": metrics["mean_r2"], "params": params, "metrics": metrics}

    # --- refit best hyperparams on FULL data for the production model ---
    media_transformed = transform_media(media_raw, best["params"]["lambdas"], best["params"]["k"], best["params"]["s"])
    X_full = np.hstack([media_transformed, controls])
    final_scaler = StandardScaler().fit(X_full)
    X_scaled = final_scaler.transform(X_full)
    final_model = ElasticNet(
        alpha=best["params"]["alpha"], l1_ratio=best["params"]["l1_ratio"], max_iter=5000
    )
    final_model.fit(X_scaled, y)

    feature_names = channel_cols + continuous_cols + binary_cols
    coefficients = dict(zip(feature_names, final_model.coef_.tolist()))

    artifact = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "channel_cols": channel_cols,
        "continuous_cols": continuous_cols,
        "binary_cols": binary_cols,
        "feature_names": feature_names,
        "best_params": {
            "lambdas": {channel_cols[i]: v for i, v in best["params"]["lambdas"].items()},
            "k": {channel_cols[i]: v for i, v in best["params"]["k"].items()},
            "s": {channel_cols[i]: v for i, v in best["params"]["s"].items()},
            "alpha": best["params"]["alpha"],
            "l1_ratio": best["params"]["l1_ratio"],
        },
        "cv_metrics": best["metrics"],
        "coefficients": coefficients,
        "intercept": float(final_model.intercept_),
        "n_weeks_trained_on": len(y),
        "model": final_model,
        "scaler": final_scaler,
    }
    return artifact, all_trials


def save_artifact(artifact: dict, config: dict) -> None:
    model_path = PROJECT_ROOT / config["paths"]["model_artifact"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    # append lightweight (JSON-safe) summary to run history, for drift_check.py
    history_path = PROJECT_ROOT / config["paths"]["run_history"]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)

    history.append({
        "trained_at": artifact["trained_at"],
        "coefficients": artifact["coefficients"],
        "cv_mean_r2": artifact["cv_metrics"]["mean_r2"],
        "cv_mean_mape": artifact["cv_metrics"]["mean_mape"],
        "best_params": artifact["best_params"],
        "n_weeks_trained_on": artifact["n_weeks_trained_on"],
    })
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    cv_report_path = PROJECT_ROOT / config["paths"]["cv_report"]
    cv_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_report_path, "w") as f:
        json.dump({
            "trained_at": artifact["trained_at"],
            "cv_metrics": artifact["cv_metrics"],
            "best_params": artifact["best_params"],
        }, f, indent=2)


if __name__ == "__main__":
    cfg = load_config()
    print("Running joint hyperparameter search (adstock + saturation + ElasticNet)...")
    art, trials = run_search(cfg)
    print(f"Best CV mean R2: {art['cv_metrics']['mean_r2']:.4f}")
    print(f"Best CV mean MAPE: {art['cv_metrics']['mean_mape']:.4f}")
    print(f"alpha={art['best_params']['alpha']:.4f}  l1_ratio={art['best_params']['l1_ratio']:.3f}")
    print("\nAdstock lambdas:", {k: round(v, 3) for k, v in art["best_params"]["lambdas"].items()})
    print("\nTop coefficients (media channels):")
    for ch in art["channel_cols"]:
        print(f"  {ch}: {art['coefficients'][ch]:.2f}")

    save_artifact(art, cfg)
    print(f"\nSaved model artifact -> {cfg['paths']['model_artifact']}")
    print(f"Saved CV report -> {cfg['paths']['cv_report']}")
