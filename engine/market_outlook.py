import pandas as pd


def build_market_outlook(features: pd.DataFrame, regime: str) -> dict:

    momentum = features["return_20"].mean()

    short_momentum = features["return_5"].mean()

    volatility = features["volatility"].mean()

    distance_high = features["distance_high"].mean()

    rel_strength = features["rel_strength"].mean()

    # -------------------------
    # SCORE MARKET CONDITIONS
    # -------------------------

    score = 0

    if momentum > 0:
        score += 1

    if short_momentum > 0:
        score += 1

    if rel_strength > 0:
        score += 1

    if volatility < features["volatility"].median():
        score += 1

    if score == 4:
        short_term = "Strong Positive"

    elif score == 3:
        short_term = "Positive"

    elif score == 2:
        short_term = "Neutral"

    elif score == 1:
        short_term = "Negative"

    else:
        short_term = "Strong Negative"

    # -------------------------
    # RISK LEVEL
    # -------------------------

    if volatility > features["volatility"].quantile(0.75):
        risk = "High"

    elif volatility < features["volatility"].quantile(0.25):
        risk = "Low"

    else:
        risk = "Medium"

    # -------------------------
    # MARKET COMMENTARY
    # -------------------------

    comments = []

    if momentum > 0:
        comments.append("Medium-term momentum is positive.")

    else:
        comments.append("Medium-term momentum is weak.")

    if rel_strength > 0:
        comments.append("Leading stocks show relative strength.")

    else:
        comments.append("Relative strength is weak.")

    if volatility < features["volatility"].median():
        comments.append("Volatility is controlled.")

    else:
        comments.append("Volatility is elevated.")

    if distance_high > 0.95:
        comments.append("Market is close to previous highs.")

    elif distance_high < 0.85:
        comments.append("Market is trading below previous highs.")

    analysis = " ".join(comments)

    return {
        "regime": regime,
        "short_term": short_term,
        "risk": risk,
        "momentum": momentum,
        "relative_strength": rel_strength,
        "volatility": volatility,
        "distance_high": distance_high,
        "analysis": analysis,
    }
