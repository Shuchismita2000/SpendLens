"""
weekly_retrain.py
=================

Automation entry point for the Marketing Mix Modeling (MMM) retraining
pipeline. This script is intended to be executed on a regular schedule
(e.g., via cron or GitHub Actions) whenever a new week's marketing
performance data becomes available.

Pipeline steps:
    1. (Optional) Simulate one new week of marketing data and append it to
       `data/raw/aurel_weekly_observed.csv`. In production, this step is
       replaced by an actual data ingestion process (e.g., warehouse query,
       GA/Shopify export, or advertising platform API).
    2. Preprocess the raw data and engineer model features.
    3. Train and tune the Marketing Mix Model, performing joint
       hyperparameter optimization and refitting the final model.
    4. Evaluate model performance using recovered-vs-true comparisons and
       holdout diagnostics.
    5. Check coefficient drift by comparing the current model against the
       previous version.
    6. Optimize and recommend next week's marketing budget allocation.

Usage:
    python scripts/weekly_retrain.py               # simulate new data + run full pipeline
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# Project Configuration
# =============================================================================

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow direct execution of this script
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config  # noqa: E402
from src.data.preprocess import preprocess  # noqa: E402
from src.models.train_model import run_search, save_artifact  # noqa: E402
from src.models.evaluate import run_evaluation  # noqa: E402
from src.monitoring.drift_check import check_drift  # noqa: E402
from src.optimization.budget_optimizer import optimize_budget  # noqa: E402


# =============================================================================
# Synthetic Data Simulation
# =============================================================================

def simulate_new_week(
    config: dict,
    n_weeks: int = 1,
    seed_offset: int = 0,
) -> None:
    """
    Simulate one or more new weeks of marketing performance data.

    This function mimics the arrival of new weekly business data by
    extending the existing dataset using realistic trends, seasonality,
    promotions, and random variation.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.
    n_weeks : int, default=1
        Number of future weeks to generate.
    seed_offset : int, default=0
        Additional random seed offset for reproducibility.

    Returns
    -------
    None
    """
    raw_path = PROJECT_ROOT / config["paths"]["raw_data"]

    df = pd.read_csv(
        raw_path,
        parse_dates=["week_start"],
    )

    last = df.iloc[-1]
    last_idx = int(last["week_idx"])

    rng = np.random.default_rng(
        1000 + last_idx + seed_offset
    )

    channel_cols = [
        channel["spend_col"]
        for channel in config["channels"]
    ]

    new_rows = []

    # -------------------------------------------------------------------------
    # Generate future weekly observations
    # -------------------------------------------------------------------------

    for step in range(1, n_weeks + 1):

        idx = last_idx + step

        date = last["week_start"] + pd.Timedelta(
            weeks=step
        )

        row = {
            "week_start": date,
            "week_idx": idx,
            "month": date.month,
            "week_of_year": int(
                date.isocalendar()[1]
            ),
        }

        # Seasonal indicators
        festive = int(
            (row["week_of_year"] in range(40, 45))
            or (row["week_of_year"] >= 50)
            or (row["week_of_year"] <= 1)
        )

        payday = int(date.day <= 7)
        sale_event = int(rng.random() < 0.10)

        promo = int(
            festive
            or sale_event
            or (rng.random() < 0.10)
        )

        row.update(
            {
                "festive_flag": festive,
                "payday_week": payday,
                "sale_event_flag": sale_event,
                "promo_flag": promo,
            }
        )

        # Pricing variables
        discount = (
            max(0.0, rng.normal(3, 1.5))
            + (rng.uniform(12, 25) if promo else 0)
        )

        discount = round(min(discount, 35), 1)

        price = round(
            last["avg_price"]
            * (1 + rng.normal(0.001, 0.01)),
            0,
        )

        row.update(
            {
                "discount_rate": discount,
                "avg_price": price,
                "coupon_usage_pct": round(
                    min(
                        discount * rng.uniform(1.2, 1.8),
                        80,
                    ),
                    1,
                ),
            }
        )

        # Marketing spend simulation
        for channel, column in zip(
            [c["name"] for c in config["channels"]],
            channel_cols,
        ):

            growth = 1 + rng.normal(
                0.002,
                0.01,
            )

            row[column] = max(
                0.0,
                round(
                    float(last[column])
                    * growth
                    * rng.lognormal(0, 0.12),
                    0,
                ),
            )

        # Market indicators
        row["consumer_confidence"] = round(
            float(
                np.clip(
                    last["consumer_confidence"]
                    + rng.normal(0, 1.8),
                    60,
                    140,
                )
            ),
            1,
        )

        row["category_search_index"] = round(
            float(
                np.clip(
                    last["category_search_index"]
                    + rng.normal(0, 2.2),
                    60,
                    140,
                )
            ),
            1,
        )

        row["stock_out_flag"] = int(
            rng.random() < 0.05
        )

        row["delivery_delay_flag"] = int(
            rng.random() < 0.08
        )

        # ---------------------------------------------------------------------
        # Simulated business performance
        # ---------------------------------------------------------------------

        base_units = float(
            last.get("units_sold", 7500)
        )

        units = max(
            500.0,
            base_units
            * (1 + rng.normal(0.003, 0.05))
            - (
                row["stock_out_flag"]
                * base_units
                * 0.20
            ),
        )

        row["units_sold"] = round(units, 0)
        row["orders"] = round(units / 1.15, 0)

        returns_pct = round(
            4.0
            + rng.normal(0, 0.6)
            + (
                5.5
                if row["delivery_delay_flag"]
                else 0
            ),
            1,
        )

        row["returns_pct"] = returns_pct

        effective_price = (
            price
            * (1 - discount / 100)
        )

        gross_revenue = (
            units * effective_price
        )

        row["gross_revenue"] = round(
            gross_revenue,
            0,
        )

        row["revenue"] = round(
            gross_revenue
            * (1 - returns_pct / 100),
            0,
        )

        row["avg_order_value"] = round(
            row["revenue"]
            / max(1, row["orders"]),
            0,
        )

        new_rows.append(row)

    # Append new observations
    new_df = pd.DataFrame(new_rows)

    combined = pd.concat(
        [df, new_df],
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        raw_path,
        index=False,
    )

    print(
        f"Simulated {n_weeks} new week(s) "
        f"-> {raw_path} "
        f"(now {len(combined)} weeks total)"
    )


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """
    Execute the complete model retraining workflow.

    Steps
    -----
    1. Simulate new data (optional).
    2. Train the model.
    3. Evaluate recovery metrics.
    4. Detect coefficient drift.
    5. Optimize marketing budget.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help="Skip synthetic data generation.",
    )

    parser.add_argument(
        "--weeks",
        type=int,
        default=1,
        help="Number of weeks to simulate.",
    )

    args = parser.parse_args()

    cfg = load_config()

    # -------------------------------------------------------------------------
    # Optional simulation
    # -------------------------------------------------------------------------

    if not args.no_simulate:
        simulate_new_week(
            cfg,
            n_weeks=args.weeks,
        )

    # -------------------------------------------------------------------------
    # Model Training
    # -------------------------------------------------------------------------

    print(
        "\n[1/4] Training "
        "(preprocess → features → hyperparameter search)..."
    )

    artifact, _ = run_search(cfg)

    save_artifact(
        artifact,
        cfg,
    )

    print(
        f"CV Mean R²: "
        f"{artifact['cv_metrics']['mean_r2']:.3f}"
    )

    print(
        f"CV Mean MAPE: "
        f"{artifact['cv_metrics']['mean_mape']:.3f}"
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    print(
        "\n[2/4] Evaluating..."
    )

    eval_report = run_evaluation(cfg)

    print(
        f"Adstock MAE: "
        f"{eval_report['adstock_mean_abs_error']:.3f}"
    )

    print(
        "Saturation Shape Correlation: "
        f"{eval_report['saturation_mean_shape_correlation']:.3f}"
    )

    # -------------------------------------------------------------------------
    # Drift Detection
    # -------------------------------------------------------------------------

    print(
        "\n[3/4] Checking Model Drift..."
    )

    drift_report = check_drift(cfg)

    if drift_report["status"] == "insufficient_history":

        print(
            drift_report["message"]
        )

    else:

        print(
            f"Flagged Channels: "
            f"{drift_report['n_channels_flagged']}"
        )

        for flag in drift_report["flags"]:

            if flag["flagged"]:

                print(
                    f"⚠ {flag['channel']}: "
                    f"{flag['pct_change']:+.1f}%"
                )

    # -------------------------------------------------------------------------
    # Budget Optimization
    # -------------------------------------------------------------------------

    print(
        "\n[4/4] Optimizing Marketing Budget..."
    )

    model_path = (
        PROJECT_ROOT
        / cfg["paths"]["model_artifact"]
    )

    artifact = joblib.load(model_path)

    optimization = optimize_budget(
        artifact,
        cfg,
    )

    print(
        "Expected Lift vs. Even Split: "
        f"{optimization['expected_lift_vs_even_split_pct']:.1f}%"
    )

    print(
        "\nRetraining complete."
    )

    print(
        "Reports saved to outputs/reports/."
    )


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    main()