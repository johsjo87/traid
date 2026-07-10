import pandas as pd


def calculate_reliability(samples, positive_rate):
    """
    Räknar hur mycket vi kan lita på historiken.
    """

    if samples < 20:
        return "LOW"

    if samples < 50:
        return "MEDIUM"

    if samples >= 100 and positive_rate >= 60:
        return "HIGH"

    return "MEDIUM"


def calculate_setup_quality(avg_return, positive_rate, samples):
    """
    Kombinerar:
    - framtida avkastning
    - träffsäkerhet
    - mängd historik
    """

    score = 50

    # avkastning
    if avg_return > 0.05:
        score += 20

    elif avg_return > 0.02:
        score += 10

    # positiv historik

    if positive_rate > 70:
        score += 20

    elif positive_rate > 60:
        score += 10

    # antal exempel

    if samples > 100:
        score += 10

    return min(100, score)


def predict_from_history(
    current_features: pd.DataFrame,
    history: pd.DataFrame,
    tolerance: float = 0.03,
):

    predictions = []

    for _, current in current_features.iterrows():
        similar = history.copy()

        similar = similar[
            (similar["return_20"] - current["return_20"]).abs() < tolerance
        ]

        similar = similar[
            (similar["rel_strength"] - current["rel_strength"]).abs() < tolerance
        ]

        similar = similar[(similar["volatility"] - current["volatility"]).abs() < 0.01]

        samples = len(similar)

        if samples == 0:
            predictions.append(
                {
                    "symbol": current["symbol"],
                    "setup_quality": 0,
                    "avg_future_return": None,
                    "positive_rate": None,
                    "reliability": "LOW",
                    "samples": 0,
                }
            )

            continue

        avg_return = similar["future_return"].mean()

        positive_rate = (similar["future_return"] > 0).mean() * 100

        quality = calculate_setup_quality(avg_return, positive_rate, samples)

        reliability = calculate_reliability(samples, positive_rate)

        predictions.append(
            {
                "symbol": current["symbol"],
                "setup_quality": quality,
                "avg_future_return": avg_return,
                "positive_rate": positive_rate,
                "reliability": reliability,
                "samples": samples,
            }
        )

    return pd.DataFrame(predictions)
