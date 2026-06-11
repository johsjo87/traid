from engine.risk import apply_risk_adjustment
import pandas as pd


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
    df["z_trend"] = _zscore(df, "trend_strength")
    df["z_vol"] = _zscore(df, "volatility")
    df["z_dist"] = _zscore(df, "distance_high")
    df["z_rsi"] = _zscore(df, "rsi")

    # Lägre volatilitet är bättre
    df["inv_vol"] = -df["z_vol"]

    # -------------------------
    # REGIME ADAPTIVE WEIGHTS
    # -------------------------

    if regime == "BULL":
        df["alpha"] = (
            0.20 * df["z_mom5"]
            + 0.30 * df["z_mom20"]
            + 0.25 * df["z_trend"]
            + 0.10 * df["inv_vol"]
            + 0.05 * df["z_dist"]
            + 0.10 * df["z_rsi"]
        )

    elif regime == "BEAR":
        df["alpha"] = (
            0.10 * df["z_mom5"]
            + 0.15 * df["z_mom20"]
            + 0.20 * df["z_trend"]
            + 0.35 * df["inv_vol"]
            + 0.10 * df["z_dist"]
            + 0.10 * df["z_rsi"]
        )

    else:
        df["alpha"] = (
            0.15 * df["z_mom5"]
            + 0.25 * df["z_mom20"]
            + 0.25 * df["z_trend"]
            + 0.15 * df["inv_vol"]
            + 0.10 * df["z_dist"]
            + 0.10 * df["z_rsi"]
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
