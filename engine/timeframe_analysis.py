import pandas as pd


def analyze_timeframes(row: pd.Series) -> dict:

    short_score = 0
    medium_score = 0
    long_score = 0

    # SHORT TERM
    if row["return_5"] > 0:
        short_score += 1

    if row["volatility"] < 0.03:
        short_score += 1

    # MEDIUM TERM
    if row["return_20"] > 0:
        medium_score += 1

    if row["rel_strength"] > 0:
        medium_score += 1

    # LONG TERM
    if row["trend_strength"] > 1:
        long_score += 1

    if row["distance_high"] > 0.85:
        long_score += 1

    def classify(score):

        if score >= 2:
            return "Positive"

        elif score == 1:
            return "Neutral"

        else:
            return "Negative"

    return {
        "short_term": classify(short_score),
        "medium_term": classify(medium_score),
        "long_term": classify(long_score),
    }


def analyze_timeframes_portfolio(portfolio: pd.DataFrame, features: pd.DataFrame):

    merged = portfolio.merge(features, on="symbol", how="left")

    results = []

    for _, row in merged.iterrows():
        result = analyze_timeframes(row)

        result["symbol"] = row["symbol"]

        results.append(result)

    return results
