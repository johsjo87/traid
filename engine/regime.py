import pandas as pd


def detect_regime(df: pd.DataFrame) -> str:

    df = df.copy()

    positive_momentum = (df["return_20"] > 0).mean()

    strong_trend = (df["trend_strength"] > 1.02).mean()

    near_high = (df["distance_high"] > 0.95).mean()

    volatility = df["volatility"].mean()

    normal_vol = df["volatility"].median()

    # volatilitet ska påverka mindre
    if volatility > normal_vol * 1.5:
        volatility_penalty = 0.15
    else:
        volatility_penalty = 0.05

    score = (
        positive_momentum * 0.45
        + strong_trend * 0.35
        + near_high * 0.20
        - volatility_penalty
    )

    if score > 0.60:
        return "BULL"

    elif score < 0.35:
        return "BEAR"

    else:
        return "NEUTRAL"
