"""
weekly_retrain.py
==================
The automation entrypoint: what a cron job / GitHub Actions workflow would
call every week when a new week's performance data lands.

Steps:
    1. (demo only) simulate one new week of raw data landing, appended to
       data/raw/aurel_weekly_observed.csv -- in production this step is
       replaced by an actual data pull (warehouse query, GA/Shopify export,
       ad platform API, etc.), everything downstream is unchanged.
    2. preprocess -> build_features -> train_model (joint hyperparameter
       search + refit)
    3. evaluate (recovered-vs-true + holdout diagnostics)
    4. drift_check (compare this run's coefficients to last run's)
    5. budget_optimizer (recommend next week's allocation)

Run:
    python scripts/weekly_retrain.py             # simulate + full pipeline
    python scripts/weekly_retrain.py --no-simulate  # just rerun on existing data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config  # noqa: E402
from src.data.preprocess import preprocess  # noqa: E402
from src.models.train_model import run_search, save_artifact  # noqa: E402
from src.models.evaluate import run_evaluation  # noqa: E402
from src.monitoring.drift_check import check_drift  # noqa: E402
from src.optimization.budget_optimizer import optimize_budget  # noqa: E402

import joblib


def simulate_new_week(config: dict, n_weeks: int = 1, seed_offset: int = 0) -> None:
    """
    Appends n_weeks of new synthetic weekly rows to the raw CSV, continuing
    trend/seasonality/spend patterns from the last known week. Stands in for
    a real data pull in production.
    """
    raw_path = PROJECT_ROOT / config["paths"]["raw_data"]
    df = pd.read_csv(raw_path, parse_dates=["week_start"])
    last = df.iloc[-1]
    last_idx = int(last["week_idx"])
    rng = np.random.default_rng(1000 + last_idx + seed_offset)

    channel_cols = [c["spend_col"] for c in config["channels"]]
    new_rows = []
    for step in range(1, n_weeks + 1):
        idx = last_idx + step
        date = last["week_start"] + pd.Timedelta(weeks=step)
        row = {"week_start": date, "week_idx": idx, "month": date.month, "week_of_year": int(date.isocalendar()[1])}

        festive = int((row["week_of_year"] in range(40, 45)) or (row["week_of_year"] >= 50) or (row["week_of_year"] <= 1))
        payday = int(date.day <= 7)
        sale_event = int(rng.random() < 0.10)
        promo = int(festive or sale_event or (rng.random() < 0.10))
        row.update({"festive_flag": festive, "payday_week": payday, "sale_event_flag": sale_event, "promo_flag": promo})

        discount = max(0.0, rng.normal(3, 1.5)) + (rng.uniform(12, 25) if promo else 0)
        discount = round(min(discount, 35), 1)
        price = round(last["avg_price"] * (1 + rng.normal(0.001, 0.01)), 0)
        row.update({"discount_rate": discount, "avg_price": price,
                    "coupon_usage_pct": round(min(discount * rng.uniform(1.2, 1.8), 80), 1)})

        for ch, col in zip([c["name"] for c in config["channels"]], channel_cols):
            growth = 1 + rng.normal(0.002, 0.01)
            row[col] = max(0.0, round(float(last[col]) * growth * rng.lognormal(0, 0.12), 0))

        row["consumer_confidence"] = round(float(np.clip(last["consumer_confidence"] + rng.normal(0, 1.8), 60, 140)), 1)
        row["category_search_index"] = round(float(np.clip(last["category_search_index"] + rng.normal(0, 2.2), 60, 140)), 1)
        row["stock_out_flag"] = int(rng.random() < 0.05)
        row["delivery_delay_flag"] = int(rng.random() < 0.08)

        # simple demand proxy for the simulated week (not the full ground-truth
        # generator -- good enough to exercise the retrain pipeline end-to-end)
        base_units = float(last.get("units_sold", 7500))
        units = max(500.0, base_units * (1 + rng.normal(0.003, 0.05)) - (row["stock_out_flag"] * base_units * 0.2))
        row["units_sold"] = round(units, 0)
        row["orders"] = round(units / 1.15, 0)
        returns_pct = round(4.0 + rng.normal(0, 0.6) + (5.5 if row["delivery_delay_flag"] else 0), 1)
        row["returns_pct"] = returns_pct
        eff_price = price * (1 - discount / 100)
        gross_rev = units * eff_price
        row["gross_revenue"] = round(gross_rev, 0)
        row["revenue"] = round(gross_rev * (1 - returns_pct / 100), 0)
        row["avg_order_value"] = round(row["revenue"] / max(1, row["orders"]), 0)

        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True, sort=False)
    combined.to_csv(raw_path, index=False)
    print(f"Simulated {n_weeks} new week(s) -> {raw_path} (now {len(combined)} weeks total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-simulate", action="store_true", help="Skip simulating a new week; use existing raw data as-is.")
    parser.add_argument("--weeks", type=int, default=1, help="Number of new weeks to simulate.")
    args = parser.parse_args()

    cfg = load_config()

    if not args.no_simulate:
        simulate_new_week(cfg, n_weeks=args.weeks)

    print("\n[1/4] Training (preprocess -> features -> joint hyperparameter search)...")
    artifact, _ = run_search(cfg)
    save_artifact(artifact, cfg)
    print(f"  CV mean R2={artifact['cv_metrics']['mean_r2']:.3f}  MAPE={artifact['cv_metrics']['mean_mape']:.3f}")

    print("\n[2/4] Evaluating (recovered-vs-true, holdout diagnostics)...")
    eval_report = run_evaluation(cfg)
    print(f"  Adstock mean abs error: {eval_report['adstock_mean_abs_error']:.3f}")
    print(f"  Saturation mean shape correlation: {eval_report['saturation_mean_shape_correlation']:.3f}")

    print("\n[3/4] Drift check (vs. previous run)...")
    drift_report = check_drift(cfg)
    if drift_report["status"] == "insufficient_history":
        print(f"  {drift_report['message']}")
    else:
        print(f"  Flagged channels: {drift_report['n_channels_flagged']}")
        for f in drift_report["flags"]:
            if f["flagged"]:
                print(f"    \u26a0 {f['channel']}: {f['pct_change']:+.1f}%")

    print("\n[4/4] Re-optimizing next week's budget allocation...")
    model_path = PROJECT_ROOT / cfg["paths"]["model_artifact"]
    art = joblib.load(model_path)
    opt_report = optimize_budget(art, cfg)
    print(f"  Expected lift vs. even-split baseline: {opt_report['expected_lift_vs_even_split_pct']:.1f}%")

    print("\nRetrain complete. All reports written to outputs/reports/.")


if __name__ == "__main__":
    main()
