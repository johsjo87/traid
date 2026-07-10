import pandas as pd

from engine.risk import apply_risk_adjustment
from engine.model_config import FEATURE_WEIGHTS


def _zscore(df: pd.DataFrame, col: str):

    std = df[col].std()

    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=df.index)

    return (df[col] - df[col].mean()) / (std + 1e-9)


def build_alpha(features: pd.DataFrame, regime: str) -> pd.DataFrame:

    df = features.copy()

    # -------------------------
    # NORMALIZED FEATURES
    # -------------------------

    df["z_mom5"] = _zscore(df, "return_5")

    df["z_mom20"] = _zscore(df, "return_20")

    df["z_vol"] = _zscore(df, "volatility")

    df["z_rel"] = _zscore(df, "rel_strength")

    # -------------------------
    # DATA DRIVEN ALPHA
    # -------------------------

    df["alpha"] = (
        FEATURE_WEIGHTS["return_20"] * df["z_mom20"]
        + FEATURE_WEIGHTS["return_5"] * df["z_mom5"]
        + FEATURE_WEIGHTS["rel_strength"] * df["z_rel"]
        - FEATURE_WEIGHTS["volatility"] * df["z_vol"]
    )

    # -------------------------
    # SCORE 0-100
    # -------------------------

    alpha_min = df["alpha"].min()
    alpha_max = df["alpha"].max()

    if abs(alpha_max - alpha_min) < 1e-9:
        df["score"] = 50.0

    else:
        df["score"] = (df["alpha"] - alpha_min) / (alpha_max - alpha_min) * 100

    # -------------------------
    # RISK ADJUSTMENT
    # -------------------------

    df = apply_risk_adjustment(df)

    # -------------------------
    # SIGNALS
    # -------------------------

    df["signal"] = "HOLD"

    df.loc[df["score"] >= 70, "signal"] = "BUY"

    df.loc[df["score"] <= 30, "signal"] = "SELL"

    return df.sort_values("score", ascending=False).reset_index(drop=True)
