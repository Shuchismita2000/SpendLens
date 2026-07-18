"""
drift_check.py
===============
Compares the latest model run's coefficients against the immediately prior
run (both stored in outputs/model_artifacts/run_history.json by
train_model.py). Flags any channel whose coefficient moved more than
`drift.coefficient_pct_change_threshold` (default 25%) week-over-week.

This is a PRODUCT decision, not a modeling technique: a marketer who sees
an automated model silently change its mind about a channel loses trust in
it. Surfacing "Meta's coefficient just jumped 40%, verify before acting"
is what makes weekly automated retraining safe to hand to a non-technical
user without a data scientist reviewing every run.

Run:
    python src/monitoring/drift_check.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.process.load_data import load_config  # noqa: E402


def check_drift(config: dict) -> dict:
    history_path = PROJECT_ROOT / config["paths"]["run_history"]
    with open(history_path, "r") as f:
        history = json.load(f)

    threshold = config["drift"]["coefficient_pct_change_threshold"]
    channel_cols = [c["spend_col"] for c in config["channels"]]

    if len(history) < 2:
        return {
            "status": "insufficient_history",
            "message": "Need at least 2 training runs to check drift. Run train_model.py again after new data lands.",
            "flags": [],
        }

    prev_run = history[-2]
    curr_run = history[-1]

    flags = []
    # Scoped to media CHANNEL coefficients only -- that's what the problem
    # statement means by "a channel's coefficient swings too much." Control
    # features (seasonality encodings, price, etc.) are excluded here: their
    # coefficients often sit near zero, so tiny absolute changes produce
    # enormous, meaningless percentage swings and would just add noise to
    # the trust signal a marketer actually needs to see.
    for channel in channel_cols:
        curr_coef = curr_run["coefficients"].get(channel)
        prev_coef = prev_run["coefficients"].get(channel)
        if curr_coef is None or prev_coef is None or prev_coef == 0:
            continue
        pct_change = (curr_coef - prev_coef) / abs(prev_coef)
        flagged = abs(pct_change) > threshold
        flags.append({
            "channel": channel,
            "prev_coefficient": round(prev_coef, 2),
            "curr_coefficient": round(curr_coef, 2),
            "pct_change": round(pct_change * 100, 1),
            "flagged": flagged,
            "flag_level": "red" if abs(pct_change) > threshold * 1.6 else ("yellow" if flagged else "green"),
        })

    n_flagged = sum(1 for f in flags if f["flagged"])
    report = {
        "status": "ok",
        "prev_run_trained_at": prev_run["trained_at"],
        "curr_run_trained_at": curr_run["trained_at"],
        "threshold_pct": threshold * 100,
        "n_channels_flagged": n_flagged,
        "flags": sorted(flags, key=lambda f: -abs(f["pct_change"])),
    }

    out_path = PROJECT_ROOT / config["paths"]["drift_report"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    cfg = load_config()
    rep = check_drift(cfg)
    if rep["status"] == "insufficient_history":
        print(rep["message"])
    else:
        print(f"Drift check: {rep['prev_run_trained_at']} -> {rep['curr_run_trained_at']}")
        print(f"Threshold: +/-{rep['threshold_pct']:.0f}%  |  Flagged channels: {rep['n_channels_flagged']}\n")
        for f in rep["flags"]:
            icon = {"green": "  ", "yellow": "\u26a0 ", "red": "\U0001f6a8"}[f["flag_level"]]
            print(f"{icon} {f['channel']:20s} {f['prev_coefficient']:>10.2f} -> {f['curr_coefficient']:>10.2f}  ({f['pct_change']:+.1f}%)")
