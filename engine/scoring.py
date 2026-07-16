import pandas as pd


def calculate_confidence(samples: int) -> float:
    """
    Räknar hur mycket vi litar på prediction.

    Fler historiska exempel = högre förtroende.
    """

    if samples < 5:
        return 0.10

    if samples < 20:
        return 0.30

    if samples < 50:
        return 0.60

    if samples < 100:
        return 0.80

    return 1.00


def combine_scores(
    alpha: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kombinerar Alpha Score och Prediction Score.

    Alpha = teknisk analys
    Prediction = historiska liknande situationer

    Prediction justeras efter hur mycket historik
    som finns bakom signalen.
    """

    df = alpha.copy()

    prediction_data = predictions.set_index("symbol")

    df["prediction_score"] = (
        df["symbol"].map(prediction_data["prediction_score"]).fillna(0.0)
    )

    df["prediction_samples"] = df["symbol"].map(prediction_data["samples"]).fillna(0)

    # -------------------------
    # CONFIDENCE ADJUSTMENT
    # -------------------------

    df["confidence"] = df["prediction_samples"].apply(calculate_confidence)

    df["raw_prediction_bonus"] = df["prediction_score"].clip(-10, 10)

    df["prediction_bonus"] = df["raw_prediction_bonus"] * df["confidence"]

    # -------------------------
    # FINAL SCORE
    # -------------------------

    df["alpha_score"] = df["score"]

    df["final_score"] = df["alpha_score"] + df["prediction_bonus"]

    df["final_score"] = df["final_score"].clip(0, 100)

    df["score"] = df["final_score"]

    return df.sort_values("score", ascending=False).reset_index(drop=True)
