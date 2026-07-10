import pandas as pd


def analyze_stock(row: pd.Series) -> dict:

    strengths = []
    risks = []

    return_20 = row.get("return_20", 0)
    return_5 = row.get("return_5", 0)
    rel_strength = row.get("rel_strength", 0)
    volatility = row.get("volatility", 0)
    trend_strength = row.get("trend_strength", 0)

    if return_20 > 0:
        strengths.append("Positive 20 day momentum")
    else:
        risks.append("Weak 20 day momentum")

    if return_5 > 0:
        strengths.append("Positive short term momentum")
    else:
        risks.append("Weak short term momentum")

    if rel_strength > 0:
        strengths.append("Strong relative strength")
    else:
        risks.append("Weak relative strength")

    if volatility < 0.025:
        strengths.append("Controlled volatility")
    else:
        risks.append("Elevated volatility")

    if trend_strength > 1:
        strengths.append("Positive trend")
    else:
        risks.append("Weak trend")

    score = len(strengths) - len(risks)

    if score >= 3:
        outlook = "Positive"

    elif score >= 1:
        outlook = "Neutral Positive"

    elif score <= -2:
        outlook = "Negative"

    else:
        outlook = "Neutral"

    return {
        "symbol": row["symbol"],
        "score": row.get("score"),
        "signal": row.get("signal"),
        "strengths": strengths,
        "risks": risks,
        "outlook": outlook,
    }


def analyze_portfolio(portfolio: pd.DataFrame, features: pd.DataFrame) -> list:

    merged = portfolio.merge(features, on="symbol", how="left")

    results = []

    for _, row in merged.iterrows():
        results.append(analyze_stock(row))

    return results
