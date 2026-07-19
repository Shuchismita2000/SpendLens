"""
build_features.py
=================

Deterministic feature engineering for the Marketing Mix Model (MMM),
executed after `preprocess.py`.

This module prepares the standardized feature matrix by:

1. Creating cyclical (sin/cos) encodings of month and week-of-year so a
   linear model correctly captures seasonal continuity (e.g., December and
   January are adjacent rather than 11 units apart).
2. Selecting marketing channel features.
3. Selecting the final set of control variables defined in
   `configs/model_config.yaml`. Mediator variables (e.g.,
   `website_sessions` and `branded_search_index`) are intentionally
   excluded, as described in `generate_dataset.py`.
4. Separating features, target, and metadata into a standardized feature
   bundle for downstream modeling.

Adstock and Hill saturation transformations are **not** applied here.
Instead, they are implemented within the modeling pipeline
(`src/models/train_model.py`) so their parameters (adstock decay λ and
Hill saturation parameters *k* and *s*) can be jointly optimized with the
ElasticNet regularization during hyperparameter tuning. Applying these
transformations during feature engineering would fix their values before
model selection begins.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Project Configuration
# =============================================================================

# Resolve the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing project modules when running this file directly
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config  # noqa: E402


# =============================================================================
# Time-Based Feature Engineering
# =============================================================================

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical representations of month and week features.

    Time variables such as months and weeks are cyclical in nature.
    Encoding them using sine and cosine transformations preserves
    their periodic relationships (e.g., December is close to January).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing 'month' and 'week_of_year' columns.

    Returns
    -------
    pd.DataFrame
        Dataset with additional cyclical time features.
    """
    df = df.copy()

    # Encode month as cyclical features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Encode week of year as cyclical features
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


# =============================================================================
# Feature Matrix Construction
# =============================================================================

def build_feature_matrix(
    processed_df: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Construct the model-ready feature matrix.

    The function:
    - Adds cyclical time features.
    - Selects marketing channel variables.
    - Selects control variables.
    - Extracts the target variable.
    - Preserves metadata for reporting and visualization.

    Parameters
    ----------
    processed_df : pd.DataFrame
        Preprocessed dataset.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Dictionary containing:

        - ``X`` : Feature DataFrame
        - ``y`` : Target Series
        - ``channel_cols`` : Marketing spend columns
        - ``control_cols`` : Control variable columns
        - ``meta`` : Metadata (week_start, week_idx)

    Raises
    ------
    ValueError
        If any required feature or target columns are missing.
    """
    # Create cyclical calendar features
    df = add_cyclical_features(processed_df)

    # Marketing channel spend columns
    channel_cols = [
        channel["spend_col"]
        for channel in config["channels"]
    ]

    # Continuous and binary control variables
    control_cols = (
        config["controls"]["continuous"]
        + config["controls"]["binary"]
    )

    # Target variable
    target_col = config["target"]["column"]

    # Verify all required columns exist
    missing = [
        col
        for col in channel_cols + control_cols + [target_col]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"build_features: missing expected columns: {missing}"
        )

    # Construct feature matrix
    X = df[channel_cols + control_cols].reset_index(drop=True)

    # Target vector
    y = df[target_col].reset_index(drop=True)

    # Preserve metadata for later reporting
    meta = df[
        ["week_start", "week_idx"]
    ].reset_index(drop=True)

    return {
        "X": X,
        "y": y,
        "channel_cols": channel_cols,
        "control_cols": control_cols,
        "meta": meta,
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Load project configuration
    cfg = load_config()

    # Load processed dataset
    processed = pd.read_csv(
        PROJECT_ROOT / cfg["paths"]["processed_data"],
        parse_dates=["week_start"],
    )

    # Build feature bundle
    bundle = build_feature_matrix(
        processed,
        cfg,
    )

    # Display summary
    print("Feature matrix shape:", bundle["X"].shape)
    print("Channel columns:", bundle["channel_cols"])
    print("Control columns:", bundle["control_cols"])
    print(
        "Target:",
        cfg["target"]["column"],
        "| range:",
        bundle["y"].min(),
        "-",
        bundle["y"].max(),
    )