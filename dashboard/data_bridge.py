"""
data_bridge.py
===============
Single source of truth for everything the dashboard (and the chatbot) reads:
model artifact, reports, and derived per-week/per-channel contribution and
ROI series. Keeping this separate from app.py means the chatbot can import
the exact same computations instead of re-deriving numbers a second way.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_config  # noqa: E402
from src.data.preprocess import preprocess  # noqa: E402
from src.features.adstock import GeometricAdstock  # noqa: E402
from src.features.saturation import HillSaturation  # noqa: E402


def load_everything():
    cfg = load_config()
    artifact = joblib.load(PROJECT_ROOT / cfg["paths"]["model_artifact"])
    processed = preprocess(cfg)

    def _load_json(key):
        p = PROJECT_ROOT / cfg["paths"][key]
        return json.load(open(p)) if p.exists() else None

    reports = {
        "cv": _load_json("cv_report"),
        "recovery": _load_json("recovery_report"),
        "drift": _load_json("drift_report"),
        "optimizer": _load_json("optimizer_report"),
    }
    holdout_path = PROJECT_ROOT / "outputs/reports/holdout_diagnostics.json"
    reports["holdout"] = json.load(open(holdout_path)) if holdout_path.exists() else None

    history_path = PROJECT_ROOT / cfg["paths"]["run_history"]
    run_history = json.load(open(history_path)) if history_path.exists() else []

    return cfg, artifact, processed, reports, run_history


def per_channel_weekly_contribution(artifact: dict, processed: pd.DataFrame) -> pd.DataFrame:
    """
    Decomposes predicted units_sold into per-channel media contribution for
    every historical week, using the fitted adstock/saturation/coefficients.
    Converts to revenue terms using that week's effective price so ROI can
    be computed downstream.
    """
    channel_cols = artifact["channel_cols"]
    media_raw = processed[channel_cols].values.astype(float)

    lam_by_idx = {i: artifact["best_params"]["lambdas"][c] for i, c in enumerate(channel_cols)}
    k_by_idx = {i: artifact["best_params"]["k"][c] for i, c in enumerate(channel_cols)}
    s_by_idx = {i: artifact["best_params"]["s"][c] for i, c in enumerate(channel_cols)}

    adstocked = GeometricAdstock(lambdas=lam_by_idx).fit_transform(media_raw)
    saturated = HillSaturation(k=k_by_idx, s=s_by_idx).fit_transform(adstocked)

    scaler = artifact["scaler"]
    n_media = len(channel_cols)
    raw_coefs = artifact["model"].coef_[:n_media] / scaler.scale_[:n_media]

    contrib_units = saturated * raw_coefs  # (n_weeks, n_channels), broadcasting

    eff_price = processed["avg_price"] * (1 - processed["discount_rate"] / 100)
    contrib_revenue = contrib_units * eff_price.values.reshape(-1, 1)

    out = pd.DataFrame(contrib_units, columns=[f"units_{c}" for c in channel_cols])
    for i, c in enumerate(channel_cols):
        out[f"revenue_{c}"] = contrib_revenue[:, i]
    out.insert(0, "week_start", processed["week_start"].values)
    for c in channel_cols:
        out[f"spend_{c.replace('spend_', '')}"] = processed[c].values  # keep raw spend alongside
    return out


def steady_state_response_curve(artifact: dict, channel: str, spend_grid: np.ndarray) -> np.ndarray:
    """
    "If you spent X every week indefinitely" curve: steady-state adstock is
    spend / (1 - lambda), then Hill saturation, then rescaled coefficient.
    Used for the diminishing-returns chart.
    """
    lam = artifact["best_params"]["lambdas"][channel]
    k = artifact["best_params"]["k"][channel]
    s = artifact["best_params"]["s"][channel]
    idx = artifact["channel_cols"].index(channel)
    coef = artifact["model"].coef_[idx] / artifact["scaler"].scale_[idx]

    steady_adstock = spend_grid / max(1e-6, (1 - lam))
    sat = steady_adstock ** s / (steady_adstock ** s + k ** s)
    return coef * sat


def channel_roi_summary(contrib_df: pd.DataFrame, channel_cols: list, last_n_weeks: int = 4) -> pd.DataFrame:
    recent = contrib_df.tail(last_n_weeks)
    rows = []
    for c in channel_cols:
        short = c.replace("spend_", "")
        spend = recent[f"spend_{short}"].sum()
        revenue = recent[f"revenue_{c}"].sum()
        roi = revenue / spend if spend > 0 else np.nan
        rows.append({"channel": short, "spend": spend, "attributed_revenue": revenue, "roi": roi})
    return pd.DataFrame(rows).sort_values("attributed_revenue", ascending=False)
