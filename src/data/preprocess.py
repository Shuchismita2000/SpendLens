"""
preprocess.py
=============
Cleaning + light joins, run BEFORE feature engineering:
    - enforce weekly continuity (no gaps/duplicate weeks)
    - clip impossible values (negative spend, >100% rates, etc.)
    - cross-check festive_flag against the external calendar (flags mismatches
      rather than silently trusting one source over the other)
    - persist to data/processed/modeling_dataset.csv

This is intentionally separate from feature engineering (adstock/saturation/
cyclical encoding) so the "clean data" artifact can be inspected/audited on
its own — a common real-world requirement when a marketer asks "where did
this number come from."
"""
"""
Data Preprocessing Pipeline

This module performs data quality validation and preprocessing by:
1. Validating weekly continuity.
2. Clipping impossible numeric values.
3. Cross-checking festive flags against an external calendar.
4. Saving the cleaned dataset for downstream modeling.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Project Configuration
# =============================================================================

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing project modules when running this file directly
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import (  # noqa: E402
    load_calendar_events,
    load_config,
    load_raw_data,
)


# =============================================================================
# Validation Rules
# =============================================================================

# Percentage-based columns that should always lie between 0 and 100.
RATE_COLUMNS_0_100 = [
    "discount_rate",
    "coupon_usage_pct",
    "returns_pct",
]

# Columns that cannot contain negative values.
NONNEGATIVE_COLUMNS = [
    "spend_meta",
    "spend_google",
    "spend_influencer",
    "spend_email_sms",
    "spend_affiliate",
    "spend_tv_ooh",
    "units_sold",
    "orders",
    "avg_price",
    "revenue",
]


# =============================================================================
# Weekly Data Validation
# =============================================================================

def check_weekly_continuity(df: pd.DataFrame) -> None:
    """
    Verify that observations occur exactly one week apart.

    The function detects:
    - Missing weeks
    - Duplicate weeks
    - Irregular date intervals

    Parameters
    ----------
    df : pd.DataFrame
        Weekly dataset containing a 'week_start' column.

    Raises
    ------
    ValueError
        If weekly continuity is violated.
    """
    # Compute differences between consecutive weeks
    diffs = df["week_start"].diff().dropna()

    # Identify any interval that is not exactly 7 days
    bad = diffs[diffs != pd.Timedelta(days=7)]

    if len(bad) > 0:
        raise ValueError(
            f"Non-weekly gaps or duplicate weeks detected at rows: "
            f"{bad.index.tolist()}"
        )


# =============================================================================
# Numeric Data Cleaning
# =============================================================================

def clip_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip impossible numeric values to valid ranges.

    Rules
    -----
    - Non-negative metrics are clipped at zero.
    - Percentage metrics are clipped between 0 and 100.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df = df.copy()

    # Ensure non-negative metrics never fall below zero
    for col in NONNEGATIVE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Restrict percentage metrics to [0, 100]
    for col in RATE_COLUMNS_0_100:
        if col in df.columns:
            df[col] = df[col].clip(lower=0, upper=100)

    return df


# =============================================================================
# Festive Calendar Validation
# =============================================================================

def cross_check_festive_flag(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare the internal festive flag with an external event calendar.

    A week is considered associated with a festive event if it falls
    within a configurable window around the event date.

    This function does **not** overwrite the original festive flag.
    Instead, it creates:
        - calendar_festive_window
        - festive_flag_mismatch

    allowing potential data-quality issues to be reviewed manually.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly performance dataset.
    calendar : pd.DataFrame
        Calendar containing event dates.

    Returns
    -------
    pd.DataFrame
        Dataset with additional validation columns.
    """
    df = df.copy()

    # Extract festive event dates
    festive_dates = calendar.loc[
        calendar["event_type"] == "festive",
        "event_date",
    ]

    # Initialize calendar-derived festive indicator
    df["calendar_festive_window"] = 0

    # Mark weeks occurring within the festive window
    for d in festive_dates:
        mask = (
            (df["week_start"] >= d - pd.Timedelta(days=7))
            & (df["week_start"] <= d + pd.Timedelta(days=3))
        )

        df.loc[mask, "calendar_festive_window"] = 1

    # Identify disagreements between both sources
    mismatch = (
        df["festive_flag"]
        != df["calendar_festive_window"]
    )

    df["festive_flag_mismatch"] = mismatch.astype(int)

    return df


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

def preprocess(config: dict | None = None) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Steps
    -----
    1. Load raw data.
    2. Load calendar events.
    3. Validate weekly continuity.
    4. Clip invalid numeric values.
    5. Cross-check festive flags.
    6. Save processed dataset.

    Parameters
    ----------
    config : dict | None
        Project configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.
    """
    if config is None:
        config = load_config()

    # Load input datasets
    raw = load_raw_data(config=config)
    calendar = load_calendar_events(config=config)

    # Validate data consistency
    check_weekly_continuity(raw)

    # Apply preprocessing steps
    clean = clip_impossible_values(raw)
    clean = cross_check_festive_flag(clean, calendar)

    # Save processed dataset
    out_path = PROJECT_ROOT / config["paths"]["processed_data"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clean.to_csv(out_path, index=False)

    return clean


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Load configuration
    cfg = load_config()

    # Execute preprocessing pipeline
    clean_df = preprocess(cfg)

    # Report preprocessing summary
    n_mismatch = clean_df["festive_flag_mismatch"].sum()

    print(
        f"Preprocessed {len(clean_df)} weeks "
        f"-> {cfg['paths']['processed_data']}"
    )

    print(
        f"Festive-flag / calendar mismatches: "
        f"{n_mismatch} week(s)"
    )