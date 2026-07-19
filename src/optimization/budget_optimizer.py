"""
budget_optimizer.py
===================

Optimize the allocation of a fixed marketing budget across advertising
channels using a trained Marketing Mix Model (MMM).

The optimizer estimates each channel's incremental contribution by applying:

1. Geometric Adstock
2. Hill Saturation
3. The fitted ElasticNet coefficient

It solves the following optimization problem:

    maximize   Σ coef_i * sat_i(adstock_i(spend_i))
    subject to Σ spend_i = total_budget
               min_i <= spend_i <= max_i

where channel-specific spending bounds are defined in
`configs/model_config.yaml`.

Adstock uses **continuation semantics**, meaning each channel's decay starts
from its last observed historical adstock value (`last_carry`) rather than
resetting to zero. This allows the optimizer to recommend next week's budget
while accounting for existing advertising momentum.

Because the Hill saturation function composed with a positive linear
coefficient is concave, the optimization problem is well behaved. The
budget allocation is solved efficiently using the SLSQP optimizer with
budget equality and per-channel bound constraints.

The resulting optimal allocation and expected incremental contribution are
saved as a JSON report for downstream dashboards and reporting.

Run:
    python src/optimization/budget_optimizer.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize


# =============================================================================
# Project Configuration
# =============================================================================

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow running this module directly
sys.path.insert(0, str(PROJECT_ROOT))

from src.process.load_data import load_config  # noqa: E402
from src.features.saturation import HillSaturation  # noqa: E402


# =============================================================================
# Adstock Carry-Forward Computation
# =============================================================================

def compute_last_carry(
    artifact: dict,
    config: dict,
) -> dict:
    """
    Compute the final adstock carry-over value for each marketing channel.

    The carry-over represents the remaining advertising effect after the
    most recent training week and is used as the starting point for future
    budget optimization.

    Parameters
    ----------
    artifact : dict
        Trained model artifact.
    config : dict
        Project configuration.

    Returns
    -------
    dict
        Mapping of channel names to their latest adstock carry values.
    """
    from src.process.preprocess import preprocess
    from src.features.adstock import GeometricAdstock

    # Load processed dataset
    processed = preprocess(config)

    channel_cols = artifact["channel_cols"]

    media_raw = processed[channel_cols].values.astype(float)

    # Convert stored channel names back into positional indices
    lam_by_idx = {
        i: artifact["best_params"]["lambdas"][channel]
        for i, channel in enumerate(channel_cols)
    }

    # Compute adstocked media
    adstocked = GeometricAdstock(
        lambdas=lam_by_idx
    ).fit_transform(media_raw)

    # Extract last carry value
    last_carry = {
        channel_cols[i]: float(adstocked[-1, i])
        for i in range(len(channel_cols))
    }

    return last_carry


# =============================================================================
# Contribution Prediction
# =============================================================================

def predicted_contribution(
    spend_next_week: dict,
    artifact: dict,
    last_carry: dict,
) -> dict:
    """
    Estimate incremental contribution for each marketing channel.

    Pipeline:

    1. Add proposed spend to existing adstock carry.
    2. Apply Hill saturation.
    3. Multiply by the fitted model coefficient.

    Parameters
    ----------
    spend_next_week : dict
        Proposed weekly spend by channel.
    artifact : dict
        Trained model artifact.
    last_carry : dict
        Latest adstock carry values.

    Returns
    -------
    dict
        Predicted incremental contribution for every channel.
    """
    channel_cols = artifact["channel_cols"]

    scaler = artifact["scaler"]

    n_media = len(channel_cols)

    # Undo feature standardization
    raw_coefs = (
        artifact["model"].coef_[:n_media]
        / scaler.scale_[:n_media]
    )

    contributions = {}

    for i, channel in enumerate(channel_cols):

        lam = artifact["best_params"]["lambdas"][channel]
        k = artifact["best_params"]["k"][channel]
        s = artifact["best_params"]["s"][channel]

        # Update carry-over
        new_carry = (
            spend_next_week[channel]
            + lam * last_carry[channel]
        )

        # Apply saturation
        sat_value = HillSaturation(
            k=k,
            s=s,
        ).fit_transform(
            np.array([[new_carry]])
        )[0, 0]

        # Compute contribution
        contributions[channel] = float(
            raw_coefs[i] * sat_value
        )

    return contributions


# =============================================================================
# Budget Optimization
# =============================================================================

def optimize_budget(
    artifact: dict,
    config: dict,
    total_budget: float | None = None,
) -> dict:
    """
    Optimize weekly marketing budget allocation.

    The optimization maximizes predicted incremental contribution while
    satisfying:

    - Total budget constraint.
    - Minimum spend per channel.
    - Maximum spend per channel.

    Parameters
    ----------
    artifact : dict
        Trained model artifact.
    config : dict
        Project configuration.
    total_budget : float, optional
        Override default optimization budget.

    Returns
    -------
    dict
        Optimization report.
    """
    channel_cols = artifact["channel_cols"]

    # Compute current adstock state
    last_carry = compute_last_carry(
        artifact,
        config,
    )

    scaler = artifact["scaler"]

    n_media = len(channel_cols)

    raw_coefs = (
        artifact["model"].coef_[:n_media]
        / scaler.scale_[:n_media]
    )

    # Retrieve optimized parameters
    lambdas = [
        artifact["best_params"]["lambdas"][c]
        for c in channel_cols
    ]

    ks = [
        artifact["best_params"]["k"][c]
        for c in channel_cols
    ]

    ss = [
        artifact["best_params"]["s"][c]
        for c in channel_cols
    ]

    carries = [
        last_carry[c]
        for c in channel_cols
    ]

    # Channel spending constraints
    bounds_cfg = {
        channel["spend_col"]: (
            channel["min_weekly_spend"],
            channel["max_weekly_spend"],
        )
        for channel in config["channels"]
    }

    bounds = [
        bounds_cfg[channel]
        for channel in channel_cols
    ]

    budget = (
        total_budget
        or config["optimization"]["default_weekly_budget"]
    )

    # -------------------------------------------------------------------------
    # Objective Function
    # -------------------------------------------------------------------------

    def neg_total_contribution(spend_vec):
        """
        Negative objective for scipy.optimize.minimize().
        """
        total = 0.0

        for i in range(n_media):

            new_carry = (
                spend_vec[i]
                + lambdas[i] * carries[i]
            )

            sat = (
                new_carry ** ss[i]
                / (
                    new_carry ** ss[i]
                    + ks[i] ** ss[i]
                    + 1e-9
                )
            )

            total += raw_coefs[i] * sat

        return -total

    # Initial equal allocation
    x0 = np.array(
        [
            max(
                bound[0],
                min(bound[1], budget / n_media),
            )
            for bound in bounds
        ]
    )

    # Rescale to satisfy total budget
    x0 *= budget / x0.sum()

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - budget,
        }
    ]

    # Run optimization
    result = minimize(
        neg_total_contribution,
        x0,
        method=config["optimization"]["method"],
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": config["optimization"]["max_iter"],
            "ftol": 1e-9,
        },
    )

    # Optimized spending
    optimized_spend = {
        channel_cols[i]: float(result.x[i])
        for i in range(n_media)
    }

    # Equal-split baseline
    current_spend = {
        channel: float(
            np.clip(
                budget / n_media,
                *bounds_cfg[channel],
            )
        )
        for channel in channel_cols
    }

    # Predicted contributions
    contrib_opt = predicted_contribution(
        optimized_spend,
        artifact,
        last_carry,
    )

    contrib_even = predicted_contribution(
        current_spend,
        artifact,
        last_carry,
    )

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "total_budget": budget,
        "optimized_spend": optimized_spend,
        "predicted_contribution_optimized": contrib_opt,
        "predicted_total_contribution_optimized": float(
            sum(contrib_opt.values())
        ),
        "baseline_even_split_spend": current_spend,
        "predicted_contribution_even_split": contrib_even,
        "predicted_total_contribution_even_split": float(
            sum(contrib_even.values())
        ),
        "expected_lift_vs_even_split_pct": float(
            (
                sum(contrib_opt.values())
                - sum(contrib_even.values())
            )
            / max(
                1e-9,
                abs(sum(contrib_even.values())),
            )
            * 100
        ),
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":

    # Load configuration
    cfg = load_config()

    # Load trained model
    art = joblib.load(
        PROJECT_ROOT / cfg["paths"]["model_artifact"]
    )

    # Run optimization
    out = optimize_budget(
        art,
        cfg,
    )

    print(
        f"Optimization "
        f"{'Succeeded' if out['success'] else 'FAILED'}: "
        f"{out['message']}"
    )

    print(
        f"\nTotal Budget: ₹ {out['total_budget']:,.0f}\n"
    )

    print(
        f"{'Channel':14s}"
        f"{'Optimized Spend':>18s}"
        f"{'Pred. Units':>16s}"
    )

    for channel, spend in out["optimized_spend"].items():

        print(
            f"{channel:14s}"
            f"{spend:18,.0f}"
            f"{out['predicted_contribution_optimized'][channel]:16,.1f}"
        )

    print(
        f"\nPredicted Total Incremental Units "
        f"(Optimized): "
        f"{out['predicted_total_contribution_optimized']:,.1f}"
    )

    print(
        f"Predicted Total Incremental Units "
        f"(Even Split): "
        f"{out['predicted_total_contribution_even_split']:,.1f}"
    )

    print(
        f"Expected Lift: "
        f"{out['expected_lift_vs_even_split_pct']:.1f}%"
    )

    # Save optimization report
    out_path = PROJECT_ROOT / cfg["paths"]["optimizer_report"]

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(out_path, "w") as f:
        json.dump(
            out,
            f,
            indent=2,
        )

    print(
        f"\nSaved report -> "
        f"{cfg['paths']['optimizer_report']}"
    )