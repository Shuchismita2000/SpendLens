"""
evaluate.py
===========
Two jobs:
    1. "Recovered vs. true" validation -- since our data is synthetic with
       KNOWN ground-truth adstock lambda / saturation params baked in
       (see generate_dataset.py), we can directly check whether the fitted
       pipeline recovered parameters close to the truth. This is the
       strongest credibility artifact in the whole project: it proves the
       METHOD works, not just that a model fit some noisy weekly numbers.
    2. Standard holdout diagnostics (actual vs predicted plot data, residuals)
       for the final refit-on-all-data model.

Run:
    python src/models/evaluate.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.process.load_data import load_config  # noqa: E402


def load_ground_truth(config: dict) -> dict:
    gt_path = PROJECT_ROOT / "data/raw/ground_truth_params.json"
    with open(gt_path, "r") as f:
        return json.load(f)


def compare_adstock(artifact: dict, gt: dict) -> pd.DataFrame:
    rows = []
    for ch, true_lam in gt["adstock_lambda"].items():
        spend_col = f"spend_{ch}"
        fitted_lam = artifact["best_params"]["lambdas"].get(spend_col, np.nan)
        rows.append({
            "channel": ch,
            "true_lambda": true_lam,
            "fitted_lambda": round(fitted_lam, 3),
            "abs_error": round(abs(fitted_lam - true_lam), 3),
        })
    return pd.DataFrame(rows)


def compare_saturation_shape(artifact: dict, gt: dict) -> pd.DataFrame:
    """
    Compares the fitted vs. true saturation curve SHAPE (correlation of the
    curve across a common spend grid), since absolute scale (k) isn't
    directly comparable -- ground truth k is in raw-spend units, fitted k
    is searched relative to each channel's own max observed spend, and the
    output magnitude is absorbed by the downstream ElasticNet coefficient
    (see saturation.py docstring). Shape correlation is the honest metric.
    """
    rows = []
    for ch in gt["adstock_lambda"].keys():
        spend_col = f"spend_{ch}"
        true_p = gt["saturation_units_scale"][ch]
        fitted_k = artifact["best_params"]["k"][spend_col]
        fitted_s = artifact["best_params"]["s"][spend_col]

        grid = np.linspace(1, true_p["k"] * 4, 200)
        true_curve = grid ** true_p["s"] / (grid ** true_p["s"] + true_p["k"] ** true_p["s"])
        fitted_curve = grid ** fitted_s / (grid ** fitted_s + fitted_k ** fitted_s)
        corr = np.corrcoef(true_curve, fitted_curve)[0, 1]

        rows.append({
            "channel": ch,
            "true_k": true_p["k"], "fitted_k": round(fitted_k, 0),
            "true_s": true_p["s"], "fitted_s": round(fitted_s, 2),
            "shape_correlation": round(float(corr), 3),
        })
    return pd.DataFrame(rows)


def holdout_diagnostics(artifact: dict, config: dict) -> dict:
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from src.process.preprocess import preprocess
    from src.features.build_features import build_feature_matrix
    from src.models.train_model import transform_media

    processed = preprocess(config)
    bundle = build_feature_matrix(processed, config)
    X, y = bundle["X"], bundle["y"].values

    media_raw = X[artifact["channel_cols"]].values.astype(float)
    controls = X[artifact["continuous_cols"] + artifact["binary_cols"]].values.astype(float)

    media_transformed = transform_media(
        media_raw, artifact["best_params"]["lambdas"], artifact["best_params"]["k"], artifact["best_params"]["s"]
    )
    # NOTE: best_params dicts are keyed by column name here (post-save), but
    # transform_media/GeometricAdstock expect 0-indexed column positions --
    # remap before transforming.
    lam_by_idx = {i: artifact["best_params"]["lambdas"][c] for i, c in enumerate(artifact["channel_cols"])}
    k_by_idx = {i: artifact["best_params"]["k"][c] for i, c in enumerate(artifact["channel_cols"])}
    s_by_idx = {i: artifact["best_params"]["s"][c] for i, c in enumerate(artifact["channel_cols"])}
    media_transformed = transform_media(media_raw, lam_by_idx, k_by_idx, s_by_idx)

    X_full = np.hstack([media_transformed, controls])
    X_scaled = artifact["scaler"].transform(X_full)
    preds = artifact["model"].predict(X_scaled)

    return {
        "week_start": bundle["meta"]["week_start"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": y.tolist(),
        "predicted": preds.tolist(),
        "residual": (y - preds).tolist(),
    }


def run_evaluation(config: dict) -> dict:
    model_path = PROJECT_ROOT / config["paths"]["model_artifact"]
    artifact = joblib.load(model_path)
    gt = load_ground_truth(config)

    adstock_cmp = compare_adstock(artifact, gt)
    sat_cmp = compare_saturation_shape(artifact, gt)
    diag = holdout_diagnostics(artifact, config)

    report = {
        "evaluated_at": artifact["trained_at"],
        "adstock_recovery": adstock_cmp.to_dict(orient="records"),
        "adstock_mean_abs_error": float(adstock_cmp["abs_error"].mean()),
        "saturation_shape_recovery": sat_cmp.to_dict(orient="records"),
        "saturation_mean_shape_correlation": float(sat_cmp["shape_correlation"].mean()),
        "cv_metrics": artifact["cv_metrics"],
        "in_sample_mape": float(np.mean(np.abs(np.array(diag["residual"])) / np.array(diag["actual"]))),
    }

    out_path = PROJECT_ROOT / config["paths"]["recovery_report"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # also persist the diagnostics series for the dashboard's overview chart
    diag_path = PROJECT_ROOT / "outputs/reports/holdout_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)

    return report


if __name__ == "__main__":
    cfg = load_config()
    rep = run_evaluation(cfg)
    print("=== Adstock recovery (true vs fitted lambda) ===")
    for row in rep["adstock_recovery"]:
        print(f"  {row['channel']:12s} true={row['true_lambda']:.2f}  fitted={row['fitted_lambda']:.2f}  |err|={row['abs_error']:.2f}")
    print(f"Mean abs error: {rep['adstock_mean_abs_error']:.3f}")

    print("\n=== Saturation shape recovery ===")
    for row in rep["saturation_shape_recovery"]:
        print(f"  {row['channel']:12s} shape correlation={row['shape_correlation']:.3f}")
    print(f"Mean shape correlation: {rep['saturation_mean_shape_correlation']:.3f}")

    print(f"\nCV mean R2: {rep['cv_metrics']['mean_r2']:.3f} | CV mean MAPE: {rep['cv_metrics']['mean_mape']:.3f}")
    print(f"In-sample MAPE (full refit): {rep['in_sample_mape']:.3f}")
