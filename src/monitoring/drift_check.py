"""
drift_check.py
==============

Monitor coefficient drift between consecutive Marketing Mix Model (MMM)
training runs.

This module compares the latest model's channel coefficients against those
from the previous run (stored in
`outputs/model_artifacts/run_history.json`) and flags any channel whose
coefficient changes by more than
`drift.coefficient_pct_change_threshold` (default: 25%) on a
week-over-week basis.

While coefficient drift detection is not a modeling technique, it is an
important operational safeguard. Significant changes in channel
effectiveness may indicate evolving marketing dynamics, data quality
issues, or model instability. Surfacing these changes allows marketers to
review unexpected shifts before acting on automated budget recommendations,
making routine retraining safer and more transparent.

A JSON drift report is generated for dashboard visualization and
monitoring.

Run:
    python src/monitoring/drift_check.py
"""

import json
import sys
from pathlib import Path


# =============================================================================
# Project Configuration
# =============================================================================

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing project modules when running this file directly
sys.path.insert(0, str(PROJECT_ROOT))

from src.process.load_data import load_config  # noqa: E402


# =============================================================================
# Drift Detection
# =============================================================================

def check_drift(config: dict) -> dict:
    """
    Compare model coefficients across the two most recent training runs.

    Only marketing channel coefficients are monitored because they are
    the primary business metrics of interest. Control-variable
    coefficients are intentionally excluded since small absolute values
    can produce misleadingly large percentage changes.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Drift report containing:

        - Training timestamps
        - Threshold used
        - Number of flagged channels
        - Per-channel drift statistics
    """
    # Load training history
    history_path = PROJECT_ROOT / config["paths"]["run_history"]

    with open(history_path, "r") as f:
        history = json.load(f)

    threshold = config["drift"]["coefficient_pct_change_threshold"]

    channel_cols = [
        channel["spend_col"]
        for channel in config["channels"]
    ]

    # Require at least two training runs
    if len(history) < 2:
        return {
            "status": "insufficient_history",
            "message": (
                "Need at least 2 training runs to check drift. "
                "Run train_model.py again after new data lands."
            ),
            "flags": [],
        }

    prev_run = history[-2]
    curr_run = history[-1]

    flags = []

    # -------------------------------------------------------------------------
    # Compare channel coefficients
    # -------------------------------------------------------------------------

    for channel in channel_cols:

        curr_coef = curr_run["coefficients"].get(channel)
        prev_coef = prev_run["coefficients"].get(channel)

        # Skip channels without comparable coefficients
        if (
            curr_coef is None
            or prev_coef is None
            or prev_coef == 0
        ):
            continue

        # Percentage coefficient change
        pct_change = (
            (curr_coef - prev_coef)
            / abs(prev_coef)
        )

        flagged = abs(pct_change) > threshold

        # Assign severity level
        if abs(pct_change) > threshold * 1.6:
            level = "red"
        elif flagged:
            level = "yellow"
        else:
            level = "green"

        flags.append(
            {
                "channel": channel,
                "prev_coefficient": round(prev_coef, 2),
                "curr_coefficient": round(curr_coef, 2),
                "pct_change": round(
                    pct_change * 100,
                    1,
                ),
                "flagged": flagged,
                "flag_level": level,
            }
        )

    n_flagged = sum(
        1
        for flag in flags
        if flag["flagged"]
    )

    report = {
        "status": "ok",
        "prev_run_trained_at": prev_run["trained_at"],
        "curr_run_trained_at": curr_run["trained_at"],
        "threshold_pct": threshold * 100,
        "n_channels_flagged": n_flagged,
        "flags": sorted(
            flags,
            key=lambda f: -abs(f["pct_change"]),
        ),
    }

    # Save drift report
    out_path = PROJECT_ROOT / config["paths"]["drift_report"]

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

    return report


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Load project configuration
    cfg = load_config()

    # Run drift detection
    rep = check_drift(cfg)

    if rep["status"] == "insufficient_history":
        print(rep["message"])

    else:
        print(
            f"Drift Check: "
            f"{rep['prev_run_trained_at']} "
            f"-> "
            f"{rep['curr_run_trained_at']}"
        )

        print(
            f"Threshold: +/-{rep['threshold_pct']:.0f}%"
            f" | Flagged channels: "
            f"{rep['n_channels_flagged']}\n"
        )

        # Display drift summary
        for flag in rep["flags"]:

            icon = {
                "green": "  ",
                "yellow": "⚠ ",
                "red": "🚨",
            }[flag["flag_level"]]

            print(
                f"{icon} "
                f"{flag['channel']:20s} "
                f"{flag['prev_coefficient']:>10.2f} "
                f"-> "
                f"{flag['curr_coefficient']:>10.2f} "
                f"({flag['pct_change']:+.1f}%)"
            )