"""
budget_optimizer.py
====================
Given the fitted MMM (adstock lambda, saturation k/s, ElasticNet coefficient
per channel), solve:

    maximize   sum_i  coef_i * sat_i(adstock_i(spend_i))
    subject to sum_i  spend_i == total_budget
               min_i <= spend_i <= max_i   (per configs/model_config.yaml)

Adstock here uses "continuation" semantics: each channel's decay carries
forward from its last known historical adstock value (last_carry), so the
optimizer is solving for NEXT week's spend given real momentum, not
starting every channel from zero.

Hill saturation composed with a linear coefficient is concave (for
coef_i > 0), so this is a well-behaved concave-maximization / convex
-minimization problem -- SLSQP with the budget equality constraint and
per-channel bounds converges reliably without needing a global optimizer.

Run:
    python src/optimization/budget_optimizer.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.process.load_data import load_config  # noqa: E402
from src.features.saturation import HillSaturation  # noqa: E402


def compute_last_carry(artifact: dict, config: dict) -> dict:
    """Adstock carry-forward value per channel as of the most recent trained week."""
    import pandas as pd
    from src.process.preprocess import preprocess
    from src.features.adstock import GeometricAdstock

    processed = preprocess(config)
    channel_cols = artifact["channel_cols"]
    media_raw = processed[channel_cols].values.astype(float)

    lam_by_idx = {i: artifact["best_params"]["lambdas"][c] for i, c in enumerate(channel_cols)}
    adstocked = GeometricAdstock(lambdas=lam_by_idx).fit_transform(media_raw)
    last_carry = {channel_cols[i]: float(adstocked[-1, i]) for i in range(len(channel_cols))}
    return last_carry


def predicted_contribution(spend_next_week: dict, artifact: dict, last_carry: dict) -> dict:
    """
    For each channel, apply one more adstock step from last_carry using next
    week's proposed spend, then saturation, then multiply by the fitted
    (unscaled -> re-scaled) coefficient to get predicted incremental units.

    NOTE on scale: the ElasticNet was fit on STANDARDIZED features (see
    train_model.py's StandardScaler). To get contribution in raw units for
    the optimizer/dashboard, we approximate by using coef / scaler.scale_
    for the media block -- i.e. convert the standardized-space coefficient
    back to raw-saturation-output space. This is the standard unscaling
    step for a linear model fit on scaled inputs.
    """
    channel_cols = artifact["channel_cols"]
    scaler = artifact["scaler"]
    n_media = len(channel_cols)
    raw_coefs = artifact["model"].coef_[:n_media] / scaler.scale_[:n_media]

    contributions = {}
    for i, ch in enumerate(channel_cols):
        lam = artifact["best_params"]["lambdas"][ch]
        k = artifact["best_params"]["k"][ch]
        s = artifact["best_params"]["s"][ch]
        new_carry = spend_next_week[ch] + lam * last_carry[ch]
        sat_val = HillSaturation(k=k, s=s).fit_transform(np.array([[new_carry]]))[0, 0]
        contributions[ch] = float(raw_coefs[i] * sat_val)
    return contributions


def optimize_budget(artifact: dict, config: dict, total_budget: float | None = None) -> dict:
    channel_cols = artifact["channel_cols"]
    last_carry = compute_last_carry(artifact, config)
    scaler = artifact["scaler"]
    n_media = len(channel_cols)
    raw_coefs = artifact["model"].coef_[:n_media] / scaler.scale_[:n_media]

    lambdas = [artifact["best_params"]["lambdas"][c] for c in channel_cols]
    ks = [artifact["best_params"]["k"][c] for c in channel_cols]
    ss = [artifact["best_params"]["s"][c] for c in channel_cols]
    carries = [last_carry[c] for c in channel_cols]

    bounds_cfg = {c["spend_col"]: (c["min_weekly_spend"], c["max_weekly_spend"]) for c in config["channels"]}
    bounds = [bounds_cfg[c] for c in channel_cols]

    budget = total_budget or config["optimization"]["default_weekly_budget"]

    def neg_total_contribution(spend_vec):
        total = 0.0
        for i in range(n_media):
            new_carry = spend_vec[i] + lambdas[i] * carries[i]
            sat_val = new_carry ** ss[i] / (new_carry ** ss[i] + ks[i] ** ss[i] + 1e-9)
            total += raw_coefs[i] * sat_val
        return -total

    x0 = np.array([max(b[0], min(b[1], budget / n_media)) for b in bounds])
    # rescale x0 to respect the budget constraint as a starting point
    x0 = x0 * (budget / x0.sum())

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - budget}]

    result = minimize(
        neg_total_contribution, x0, method=config["optimization"]["method"],
        bounds=bounds, constraints=constraints,
        options={"maxiter": config["optimization"]["max_iter"], "ftol": 1e-9},
    )

    optimized_spend = {channel_cols[i]: float(result.x[i]) for i in range(n_media)}
    current_spend = {c: float(np.clip(budget / n_media, *bounds_cfg[c])) for c in channel_cols}  # naive even-split baseline

    contrib_optimized = predicted_contribution(optimized_spend, artifact, last_carry)
    contrib_current_even_split = predicted_contribution(current_spend, artifact, last_carry)

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "total_budget": budget,
        "optimized_spend": optimized_spend,
        "predicted_contribution_optimized": contrib_optimized,
        "predicted_total_contribution_optimized": float(sum(contrib_optimized.values())),
        "baseline_even_split_spend": current_spend,
        "predicted_contribution_even_split": contrib_current_even_split,
        "predicted_total_contribution_even_split": float(sum(contrib_current_even_split.values())),
        "expected_lift_vs_even_split_pct": float(
            (sum(contrib_optimized.values()) - sum(contrib_current_even_split.values()))
            / max(1e-9, abs(sum(contrib_current_even_split.values()))) * 100
        ),
    }


if __name__ == "__main__":
    cfg = load_config()
    art = joblib.load(PROJECT_ROOT / cfg["paths"]["model_artifact"])
    out = optimize_budget(art, cfg)

    print(f"Optimization {'succeeded' if out['success'] else 'FAILED'}: {out['message']}")
    print(f"Total budget: Rs {out['total_budget']:,.0f}\n")
    print(f"{'Channel':14s} {'Optimized Spend':>16s} {'Pred. Units':>14s}")
    for ch, sp in out["optimized_spend"].items():
        print(f"{ch:14s} {sp:16,.0f} {out['predicted_contribution_optimized'][ch]:14,.1f}")
    print(f"\nPredicted total incremental units (optimized): {out['predicted_total_contribution_optimized']:,.1f}")
    print(f"Predicted total incremental units (even-split baseline): {out['predicted_total_contribution_even_split']:,.1f}")
    print(f"Expected lift vs. naive even split: {out['expected_lift_vs_even_split_pct']:.1f}%")

    out_path = PROJECT_ROOT / cfg["paths"]["optimizer_report"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {cfg['paths']['optimizer_report']}")
