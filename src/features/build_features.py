"""
build_features.py
==================
Deterministic feature engineering, run AFTER preprocess.py:
    - cyclical (sin/cos) encoding of month and week_of_year, so a linear
      model sees December and January as adjacent rather than 11 units apart
    - selects the final control set per configs/model_config.yaml
      (mediator columns like website_sessions / branded_search_index are
      deliberately excluded -- see dataset generator docstring)

Adstock and Hill saturation are NOT applied here. They live inside the
model Pipeline (src/models/train_model.py) because their parameters
(lambda, k, s) are tuned jointly with the ElasticNet penalty via the same
hyperparameter search -- doing the transform here would freeze those
parameters before the search even starts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config  # noqa: E402


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df


def build_feature_matrix(processed_df: pd.DataFrame, config: dict) -> dict:
    """
    Returns a dict with:
        'X' : full feature DataFrame (spend cols + control cols, in fixed order)
        'y' : target Series
        'channel_cols' : list of spend column names, in channel-config order
        'control_cols' : list of control column names, in config order
        'meta' : DataFrame passthrough of week_start/week_idx for reporting
    """
    df = add_cyclical_features(processed_df)

    channel_cols = [c["spend_col"] for c in config["channels"]]
    control_cols = config["controls"]["continuous"] + config["controls"]["binary"]
    target_col = config["target"]["column"]

    missing = [c for c in channel_cols + control_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"build_features: missing expected columns: {missing}")

    X = df[channel_cols + control_cols].reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)
    meta = df[["week_start", "week_idx"]].reset_index(drop=True)

    return {
        "X": X,
        "y": y,
        "channel_cols": channel_cols,
        "control_cols": control_cols,
        "meta": meta,
    }


if __name__ == "__main__":
    cfg = load_config()
    processed = pd.read_csv(PROJECT_ROOT / cfg["paths"]["processed_data"], parse_dates=["week_start"])
    bundle = build_feature_matrix(processed, cfg)
    print("Feature matrix shape:", bundle["X"].shape)
    print("Channel columns:", bundle["channel_cols"])
    print("Control columns:", bundle["control_cols"])
    print("Target:", cfg["target"]["column"], "| range:", bundle["y"].min(), "-", bundle["y"].max())
