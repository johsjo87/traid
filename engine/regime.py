import pandas as pd


def detect_regime(df: pd.DataFrame) -> str:

    df = df.copy()

    positive_momentum = (df["return_20"] > 0).mean()
    strong_trend = (df["trend_strength"] > 1.02).mean()

    high_vol = (df["volatility"] > df["volatility"].mean()).mean()

    near_high = (df["distance_high"] > 0.95).mean()

    score = (
        positive_momentum * 0.4 + strong_trend * 0.3 + near_high * 0.2 - high_vol * 0.3
    )

    if score > 0.55:
        return "BULL"
    elif score < 0.40:
        return "BEAR"
    else:
        return "NEUTRAL"
