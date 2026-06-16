import pandas as pd


def _rank(df: pd.DataFrame, col: str):
    return df[col].rank(pct=True)


def build_alpha(features: pd.DataFrame, regime: str) -> pd.DataFrame:

    if features is None or features.empty:
        return pd.DataFrame()

    df = features.copy()

    # -------------------------
    # CROSS-SECTIONAL RANKING
    # -------------------------
    df["r_mom5"] = _rank(df, "return_5")
    df["r_mom20"] = _rank(df, "return_20")
    df["r_trend"] = _rank(df, "trend_strength")
    df["r_vol"] = 1 - _rank(df, "volatility")
    df["r_dist"] = _rank(df, "distance_high")
    df["r_rsi"] = 1 - abs(df["rsi"] - 50) / 50
    df["r_rel"] = _rank(df, "rel_strength")

    # -------------------------
    # REGIME WEIGHTS
    # -------------------------
    if regime == "BULL":
        weights = {
            "r_mom5": 0.10,
            "r_mom20": 0.25,
            "r_trend": 0.20,
            "r_vol": 0.10,
            "r_dist": 0.10,
            "r_rsi": 0.10,
            "r_rel": 0.15,
        }

    elif regime == "BEAR":
        weights = {
            "r_mom5": 0.05,
            "r_mom20": 0.10,
            "r_trend": 0.20,
            "r_vol": 0.25,
            "r_dist": 0.15,
            "r_rsi": 0.15,
            "r_rel": 0.10,
        }

    else:
        weights = {
            "r_mom5": 0.10,
            "r_mom20": 0.20,
            "r_trend": 0.20,
            "r_vol": 0.15,
            "r_dist": 0.15,
            "r_rsi": 0.05,
            "r_rel": 0.15,
        }

    # -------------------------
    # FINAL SCORE
    # -------------------------
    df["alpha"] = 0

    for k, w in weights.items():
        df["alpha"] += w * df[k]

    df["score"] = (df["alpha"] * 100).clip(0, 100)

    df["signal"] = "HOLD"
    df.loc[df["score"] >= 65, "signal"] = "BUY"
    df.loc[df["score"] <= 35, "signal"] = "SELL"

    return df.sort_values("score", ascending=False).reset_index(drop=True)
