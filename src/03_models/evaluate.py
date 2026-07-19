"""
evaluate.py
===========

Evaluate the trained Marketing Mix Model (MMM) by assessing both parameter
recovery and predictive performance.

This module performs two complementary evaluations:

1. **Recovered vs. Ground Truth Validation**
   Since the training data is synthetic with known adstock decay rates and
   Hill saturation parameters (defined in `generate_dataset.py`), the fitted
   model can be directly compared against the true underlying values. This
   provides a strong validation of the modeling methodology, demonstrating
   that the pipeline can recover the mechanisms used to generate the data,
   rather than simply fitting noisy observations.

2. **Prediction Diagnostics**
   Evaluate predictive performance using holdout diagnostics, including
   actual vs. predicted comparisons, residual analysis, and overall model
   performance metrics for the final refit model.

Evaluation artifacts and summary reports are saved as JSON files for
dashboard visualization and reporting.

Run:
    python src/models/evaluate.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# Project Configuration
# =============================================================================

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing project modules when running this file directly
sys.path.insert(0, str(PROJECT_ROOT))

from src.process.load_data import load_config  # noqa: E402


# =============================================================================
# Ground Truth Loader
# =============================================================================

def load_ground_truth(config: dict) -> dict:
    """
    Load the simulated ground-truth parameters.

    These parameters are used to evaluate how accurately the model
    recovers the known adstock and saturation settings.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Ground truth parameter dictionary.
    """
    gt_path = PROJECT_ROOT / "data/raw/ground_truth_params.json"

    with open(gt_path, "r") as f:
        return json.load(f)


# =============================================================================
# Adstock Parameter Evaluation
# =============================================================================

def compare_adstock(
    artifact: dict,
    gt: dict,
) -> pd.DataFrame:
    """
    Compare fitted adstock decay rates against ground truth.

    Parameters
    ----------
    artifact : dict
        Saved model artifact.
    gt : dict
        Ground truth parameters.

    Returns
    -------
    pd.DataFrame
        Comparison table including absolute error for each channel.
    """
    rows = []

    for ch, true_lam in gt["adstock_lambda"].items():
        spend_col = f"spend_{ch}"

        fitted_lam = artifact["best_params"]["lambdas"].get(
            spend_col,
            np.nan,
        )

        rows.append(
            {
                "channel": ch,
                "true_lambda": true_lam,
                "fitted_lambda": round(fitted_lam, 3),
                "abs_error": round(abs(fitted_lam - true_lam), 3),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Saturation Curve Evaluation
# =============================================================================

def compare_saturation_shape(
    artifact: dict,
    gt: dict,
) -> pd.DataFrame:
    """
    Compare recovered Hill saturation curves with the ground truth.

    Since the half-saturation parameter (k) is estimated relative to
    observed channel spend, direct comparison is not meaningful.

    Instead, the similarity of the curve shapes is measured using the
    Pearson correlation across a common spend grid.

    Parameters
    ----------
    artifact : dict
        Saved model artifact.
    gt : dict
        Ground truth parameters.

    Returns
    -------
    pd.DataFrame
        Saturation recovery summary.
    """
    rows = []

    for ch in gt["adstock_lambda"].keys():
        spend_col = f"spend_{ch}"

        true_p = gt["saturation_units_scale"][ch]
        fitted_k = artifact["best_params"]["k"][spend_col]
        fitted_s = artifact["best_params"]["s"][spend_col]

        # Common spend grid
        grid = np.linspace(
            1,
            true_p["k"] * 4,
            200,
        )

        # Ground truth Hill curve
        true_curve = (
            grid ** true_p["s"]
            / (grid ** true_p["s"] + true_p["k"] ** true_p["s"])
        )

        # Estimated Hill curve
        fitted_curve = (
            grid ** fitted_s
            / (grid ** fitted_s + fitted_k ** fitted_s)
        )

        corr = np.corrcoef(
            true_curve,
            fitted_curve,
        )[0, 1]

        rows.append(
            {
                "channel": ch,
                "true_k": true_p["k"],
                "fitted_k": round(fitted_k, 0),
                "true_s": true_p["s"],
                "fitted_s": round(fitted_s, 2),
                "shape_correlation": round(float(corr), 3),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Prediction Diagnostics
# =============================================================================

def holdout_diagnostics(
    artifact: dict,
    config: dict,
) -> dict:
    """
    Generate prediction diagnostics for the trained model.

    The function rebuilds the feature matrix, applies the learned
    media transformations, and produces predictions together with
    residuals.

    Parameters
    ----------
    artifact : dict
        Saved model artifact.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Prediction diagnostics.
    """
    import sys as _sys

    _sys.path.insert(0, str(PROJECT_ROOT))

    from src.process.preprocess import preprocess
    from src.features.build_features import build_feature_matrix
    from src.models.train_model import transform_media

    # Load processed data
    processed = preprocess(config)

    bundle = build_feature_matrix(
        processed,
        config,
    )

    X = bundle["X"]
    y = bundle["y"].values

    # Split media and control variables
    media_raw = X[
        artifact["channel_cols"]
    ].values.astype(float)

    controls = X[
        artifact["continuous_cols"]
        + artifact["binary_cols"]
    ].values.astype(float)

    # Convert saved parameter dictionaries to column-index mappings
    lam_by_idx = {
        i: artifact["best_params"]["lambdas"][c]
        for i, c in enumerate(artifact["channel_cols"])
    }

    k_by_idx = {
        i: artifact["best_params"]["k"][c]
        for i, c in enumerate(artifact["channel_cols"])
    }

    s_by_idx = {
        i: artifact["best_params"]["s"][c]
        for i, c in enumerate(artifact["channel_cols"])
    }

    # Apply media transformations
    media_transformed = transform_media(
        media_raw,
        lam_by_idx,
        k_by_idx,
        s_by_idx,
    )

    # Combine transformed media with control variables
    X_full = np.hstack(
        [
            media_transformed,
            controls,
        ]
    )

    # Apply scaler and generate predictions
    X_scaled = artifact["scaler"].transform(X_full)

    preds = artifact["model"].predict(X_scaled)

    return {
        "week_start": bundle["meta"]["week_start"]
        .dt.strftime("%Y-%m-%d")
        .tolist(),
        "actual": y.tolist(),
        "predicted": preds.tolist(),
        "residual": (y - preds).tolist(),
    }


# =============================================================================
# Evaluation Pipeline
# =============================================================================

def run_evaluation(config: dict) -> dict:
    """
    Execute the complete evaluation pipeline.

    Steps
    -----
    1. Load trained model.
    2. Load ground truth parameters.
    3. Evaluate adstock recovery.
    4. Evaluate saturation recovery.
    5. Generate prediction diagnostics.
    6. Save evaluation reports.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Evaluation report.
    """
    model_path = PROJECT_ROOT / config["paths"]["model_artifact"]

    artifact = joblib.load(model_path)

    gt = load_ground_truth(config)

    adstock_cmp = compare_adstock(
        artifact,
        gt,
    )

    sat_cmp = compare_saturation_shape(
        artifact,
        gt,
    )

    diag = holdout_diagnostics(
        artifact,
        config,
    )

    report = {
        "evaluated_at": artifact["trained_at"],
        "adstock_recovery": adstock_cmp.to_dict(
            orient="records"
        ),
        "adstock_mean_abs_error": float(
            adstock_cmp["abs_error"].mean()
        ),
        "saturation_shape_recovery": sat_cmp.to_dict(
            orient="records"
        ),
        "saturation_mean_shape_correlation": float(
            sat_cmp["shape_correlation"].mean()
        ),
        "cv_metrics": artifact["cv_metrics"],
        "in_sample_mape": float(
            np.mean(
                np.abs(np.array(diag["residual"]))
                / np.array(diag["actual"])
            )
        ),
    }

    # Save evaluation report
    out_path = PROJECT_ROOT / config["paths"]["recovery_report"]

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # Save diagnostics for dashboard visualization
    diag_path = (
        PROJECT_ROOT
        / "outputs/reports/holdout_diagnostics.json"
    )

    with open(diag_path, "w") as f:
        json.dump(
            diag,
            f,
            indent=2,
        )

    return report


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Load project configuration
    cfg = load_config()

    # Run evaluation pipeline
    rep = run_evaluation(cfg)

    print("=== Adstock Recovery (True vs Fitted) ===")

    for row in rep["adstock_recovery"]:
        print(
            f"{row['channel']:12s} "
            f"true={row['true_lambda']:.2f}   "
            f"fitted={row['fitted_lambda']:.2f}   "
            f"|error|={row['abs_error']:.2f}"
        )

    print(
        f"Mean absolute error: "
        f"{rep['adstock_mean_abs_error']:.3f}"
    )

    print("\n=== Saturation Shape Recovery ===")

    for row in rep["saturation_shape_recovery"]:
        print(
            f"{row['channel']:12s} "
            f"shape correlation={row['shape_correlation']:.3f}"
        )

    print(
        f"Mean shape correlation: "
        f"{rep['saturation_mean_shape_correlation']:.3f}"
    )

    print(
        f"\nCV Mean R²: {rep['cv_metrics']['mean_r2']:.3f}"
        f" | CV Mean MAPE: {rep['cv_metrics']['mean_mape']:.3f}"
    )

    print(
        f"In-sample MAPE (full refit): "
        f"{rep['in_sample_mape']:.3f}"
    )