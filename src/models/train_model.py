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

# =============================================================================
# Hyperparameter Sampling
# =============================================================================

def sample_hyperparams(
    rng: np.random.Generator,
    config: dict,
    channel_cols: list,
    max_spend: dict,
):
    """
    Randomly sample a set of model hyperparameters.

    The sampled parameters include:
    - Geometric adstock decay rates.
    - Hill saturation parameters.
    - ElasticNet regularization parameters.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    config : dict
        Project configuration dictionary.
    channel_cols : list
        Marketing channel column names.
    max_spend : dict
        Maximum observed spend for each marketing channel.

    Returns
    -------
    dict
        Dictionary containing sampled hyperparameters.
    """
    hp = config["hyperparameter_search"]

    # Hyperparameter search bounds
    lam_lo, lam_hi = hp["adstock_lambda_bounds"]
    s_lo, s_hi = hp["saturation_s_bounds"]
    kf_lo, kf_hi = hp["saturation_k_frac_bounds"]
    a_lo, a_hi = hp["elasticnet_alpha_bounds_log10"]
    l1_lo, l1_hi = hp["elasticnet_l1_ratio_bounds"]

    # Sample adstock decay rates
    lambdas = {
        i: rng.uniform(lam_lo, lam_hi)
        for i in range(len(channel_cols))
    }

    # Sample Hill steepness values
    s_params = {
        i: rng.uniform(s_lo, s_hi)
        for i in range(len(channel_cols))
    }

    # Sample Hill half-saturation values
    k_params = {}

    for i, col in enumerate(channel_cols):
        k_frac = rng.uniform(kf_lo, kf_hi)
        k_params[i] = max(
            1.0,
            k_frac * max_spend[col],
        )

    # Sample ElasticNet hyperparameters
    alpha = 10 ** rng.uniform(a_lo, a_hi)
    l1_ratio = rng.uniform(l1_lo, l1_hi)

    return {
        "lambdas": lambdas,
        "s": s_params,
        "k": k_params,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
    }


# =============================================================================
# Media Transformation
# =============================================================================

def transform_media(
    X_channels: np.ndarray,
    lambdas: dict,
    k: dict,
    s: dict,
) -> np.ndarray:
    """
    Apply media transformations.

    Each marketing channel undergoes:

    1. Geometric adstock transformation.
    2. Hill saturation transformation.

    Parameters
    ----------
    X_channels : ndarray
        Raw marketing spend matrix.
    lambdas : dict
        Adstock decay rates.
    k : dict
        Hill half-saturation values.
    s : dict
        Hill steepness parameters.

    Returns
    -------
    ndarray
        Transformed media features.
    """
    adstocked = GeometricAdstock(
        lambdas=lambdas,
    ).fit_transform(X_channels)

    saturated = HillSaturation(
        k=k,
        s=s,
    ).fit_transform(adstocked)

    return saturated


# =============================================================================
# Hyperparameter Evaluation
# =============================================================================

def evaluate_hyperparams(
    media_raw: np.ndarray,
    controls: np.ndarray,
    y: np.ndarray,
    params: dict,
    cv_splits: list,
) -> dict:
    """
    Evaluate a hyperparameter configuration using
    expanding-window cross-validation.

    Parameters
    ----------
    media_raw : ndarray
        Raw marketing spend matrix.
    controls : ndarray
        Control variable matrix.
    y : ndarray
        Target variable.
    params : dict
        Sampled hyperparameters.
    cv_splits : list
        Cross-validation splits.

    Returns
    -------
    dict
        Cross-validation metrics.
    """
    # Transform marketing variables
    media_transformed = transform_media(
        media_raw,
        params["lambdas"],
        params["k"],
        params["s"],
    )

    X_full = np.hstack(
        [
            media_transformed,
            controls,
        ]
    )

    fold_r2 = []
    fold_mape = []

    # Evaluate each fold
    for train_idx, test_idx in cv_splits:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_full[train_idx]
        )

        X_test = scaler.transform(
            X_full[test_idx]
        )

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=5000,
        )

        model.fit(
            X_train,
            y_train,
        )

        preds = model.predict(X_test)

        fold_r2.append(
            r2_score(y_test, preds)
        )

        fold_mape.append(
            mean_absolute_percentage_error(
                y_test,
                preds,
            )
        )

    return {
        "mean_r2": float(np.mean(fold_r2)),
        "mean_mape": float(np.mean(fold_mape)),
        "fold_r2": fold_r2,
        "fold_mape": fold_mape,
    }


# =============================================================================
# Hyperparameter Search Pipeline
# =============================================================================

def run_search(config: dict) -> dict:
    """
    Execute joint hyperparameter optimization.

    The pipeline performs:

    1. Data preprocessing.
    2. Feature construction.
    3. Random hyperparameter search.
    4. Cross-validation.
    5. Final model refitting.
    6. Artifact generation.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    tuple
        Model artifact and trial history.
    """
    # Load and prepare data
    processed = preprocess(config)

    bundle = build_feature_matrix(
        processed,
        config,
    )

    X = bundle["X"]
    y = bundle["y"].values

    channel_cols = bundle["channel_cols"]

    continuous_cols = config["controls"]["continuous"]
    binary_cols = config["controls"]["binary"]

    media_raw = X[channel_cols].values.astype(float)

    controls = X[
        continuous_cols + binary_cols
    ].values.astype(float)

    # Maximum observed spend
    max_spend = {
        col: X[col].max()
        for col in channel_cols
    }

    # Build CV folds
    cv_cfg = config["cross_validation"]

    cv_splits = expanding_window_splits(
        len(y),
        cv_cfg["min_train_weeks"],
        cv_cfg["n_splits"],
    )

    # Initialize random search
    hp_cfg = config["hyperparameter_search"]

    rng = np.random.default_rng(
        hp_cfg["random_state"]
    )

    best = {
        "score": -np.inf,
        "params": None,
        "metrics": None,
    }

    all_trials = []

    # Evaluate random samples
    for i in range(hp_cfg["n_iter"]):

        params = sample_hyperparams(
            rng,
            config,
            channel_cols,
            max_spend,
        )

        metrics = evaluate_hyperparams(
            media_raw,
            controls,
            y,
            params,
            cv_splits,
        )

        all_trials.append(
            {
                "iter": i,
                "mean_r2": metrics["mean_r2"],
                "mean_mape": metrics["mean_mape"],
            }
        )

        if metrics["mean_r2"] > best["score"]:
            best = {
                "score": metrics["mean_r2"],
                "params": params,
                "metrics": metrics,
            }

    # -------------------------------------------------------------------------
    # Refit the best model on the full dataset
    # -------------------------------------------------------------------------

    media_transformed = transform_media(
        media_raw,
        best["params"]["lambdas"],
        best["params"]["k"],
        best["params"]["s"],
    )

    X_full = np.hstack(
        [
            media_transformed,
            controls,
        ]
    )

    final_scaler = StandardScaler().fit(X_full)

    X_scaled = final_scaler.transform(X_full)

    final_model = ElasticNet(
        alpha=best["params"]["alpha"],
        l1_ratio=best["params"]["l1_ratio"],
        max_iter=5000,
    )

    final_model.fit(
        X_scaled,
        y,
    )

    # Store feature coefficients
    feature_names = (
        channel_cols
        + continuous_cols
        + binary_cols
    )

    coefficients = dict(
        zip(
            feature_names,
            final_model.coef_.tolist(),
        )
    )

    # Assemble model artifact
    artifact = {
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "channel_cols": channel_cols,
        "continuous_cols": continuous_cols,
        "binary_cols": binary_cols,
        "feature_names": feature_names,
        "best_params": {
            "lambdas": {
                channel_cols[i]: v
                for i, v in best["params"]["lambdas"].items()
            },
            "k": {
                channel_cols[i]: v
                for i, v in best["params"]["k"].items()
            },
            "s": {
                channel_cols[i]: v
                for i, v in best["params"]["s"].items()
            },
            "alpha": best["params"]["alpha"],
            "l1_ratio": best["params"]["l1_ratio"],
        },
        "cv_metrics": best["metrics"],
        "coefficients": coefficients,
        "intercept": float(
            final_model.intercept_
        ),
        "n_weeks_trained_on": len(y),
        "model": final_model,
        "scaler": final_scaler,
    }

    return artifact, all_trials


# =============================================================================
# Artifact Persistence
# =============================================================================

def save_artifact(
    artifact: dict,
    config: dict,
) -> None:
    """
    Save the trained model and associated reports.

    The following files are created:

    - Serialized model artifact
    - Run history
    - Cross-validation report

    Parameters
    ----------
    artifact : dict
        Trained model artifact.
    config : dict
        Project configuration dictionary.
    """
    # Save serialized model
    model_path = PROJECT_ROOT / config["paths"]["model_artifact"]

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        model_path,
    )

    # Update training history
    history_path = PROJECT_ROOT / config["paths"]["run_history"]

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)

    history.append(
        {
            "trained_at": artifact["trained_at"],
            "coefficients": artifact["coefficients"],
            "cv_mean_r2": artifact["cv_metrics"]["mean_r2"],
            "cv_mean_mape": artifact["cv_metrics"]["mean_mape"],
            "best_params": artifact["best_params"],
            "n_weeks_trained_on": artifact["n_weeks_trained_on"],
        }
    )

    with open(history_path, "w") as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    # Save CV report
    cv_report_path = PROJECT_ROOT / config["paths"]["cv_report"]

    cv_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(cv_report_path, "w") as f:
        json.dump(
            {
                "trained_at": artifact["trained_at"],
                "cv_metrics": artifact["cv_metrics"],
                "best_params": artifact["best_params"],
            },
            f,
            indent=2,
        )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Load configuration
    cfg = load_config()

    print(
        "Running joint hyperparameter search "
        "(Adstock + Saturation + ElasticNet)..."
    )

    # Train model
    art, trials = run_search(cfg)

    # Display results
    print(
        f"Best CV Mean R²: "
        f"{art['cv_metrics']['mean_r2']:.4f}"
    )

    print(
        f"Best CV Mean MAPE: "
        f"{art['cv_metrics']['mean_mape']:.4f}"
    )

    print(
        f"alpha={art['best_params']['alpha']:.4f} "
        f"l1_ratio={art['best_params']['l1_ratio']:.3f}"
    )

    print(
        "\nAdstock Lambdas:",
        {
            k: round(v, 3)
            for k, v in art["best_params"]["lambdas"].items()
        },
    )

    print("\nTop Media Coefficients:")

    for ch in art["channel_cols"]:
        print(
            f"  {ch}: "
            f"{art['coefficients'][ch]:.2f}"
        )

    # Save outputs
    save_artifact(
        art,
        cfg,
    )

    print(
        f"\nSaved model artifact -> "
        f"{cfg['paths']['model_artifact']}"
    )

    print(
        f"Saved CV report -> "
        f"{cfg['paths']['cv_report']}"
    )