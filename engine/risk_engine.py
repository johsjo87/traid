import pandas as pd


def analyze_risk(features: pd.DataFrame, regime: str) -> dict:

    volatility = features["volatility"].mean()

    distance_high = features["distance_high"].mean()

    momentum = features["return_20"].mean()

    risk_score = 0

    reasons = []

    # -------------------------
    # VOLATILITY
    # -------------------------

    if volatility > features["volatility"].median():
        risk_score += 2

        reasons.append("Volatility is elevated")

    else:
        reasons.append("Volatility is controlled")

    # -------------------------
    # MARKET POSITION
    # -------------------------

    if distance_high < 0.90:
        risk_score += 1

        reasons.append("Market is below recent highs")

    else:
        reasons.append("Market is close to highs")

    # -------------------------
    # MOMENTUM
    # -------------------------

    if momentum < 0:
        risk_score += 1

        reasons.append("Momentum is weak")

    else:
        reasons.append("Momentum is positive")

    # -------------------------
    # REGIME
    # -------------------------

    if regime == "BEAR":
        risk_score += 2

        reasons.append("Market regime is bearish")

    # -------------------------
    # FINAL RISK
    # -------------------------

    if risk_score >= 4:
        risk = "High"
        exposure = "40-60%"

    elif risk_score >= 2:
        risk = "Medium"
        exposure = "60-80%"

    else:
        risk = "Low"
        exposure = "80-100%"

    return {
        "risk": risk,
        "exposure": exposure,
        "score": risk_score,
        "reasons": reasons,
    }
