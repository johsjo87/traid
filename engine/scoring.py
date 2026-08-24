import pandas as pd


def calculate_confidence(samples: int) -> float:
    if samples < 20:
        return 0.10

    if samples < 50:
        return 0.30

    if samples < 100:
        return 0.60

    return 1.00


def calculate_prediction_impact(
    prediction_score: float,
    samples: int,
) -> float:
    if samples < 20:
        return 0.25

    if samples < 50:
        return 0.50

    if samples < 100:
        return 0.75

    return 1.00


def calibrate_prediction_score(
    prediction_score: float,
) -> float:
    """
    Converts the raw prediction score into a stronger,
    non-linear decision bonus.

    Historical validation showed that:
        < -5  = clearly weak
        -5–0  = weak
        0–2   = weak positive
        2–5   = strong positive
        5+    = very strong positive
    """

    if prediction_score < -5:
        return -6.0

    if prediction_score < 0:
        return -2.0

    if prediction_score < 2:
        return 1.0

    if prediction_score < 5:
        return 4.0

    return 8.0


def combine_scores(
    alpha: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    df = alpha.copy()

    if predictions is None or predictions.empty:
        df["prediction_score"] = 0.0
        df["prediction_samples"] = 0
        df["confidence"] = 0.0
        df["raw_prediction_bonus"] = 0.0
        df["prediction_impact"] = 0.0
        df["prediction_bonus"] = 0.0
        df["alpha_score"] = df["score"]
        df["final_score"] = df["score"]

        return df.sort_values(
            "score",
            ascending=False,
        ).reset_index(drop=True)

    prediction_data = predictions.set_index("symbol")

    df["prediction_score"] = (
        df["symbol"].map(prediction_data["prediction_score"]).fillna(0.0)
    )

    df["prediction_samples"] = df["symbol"].map(prediction_data["samples"]).fillna(0)

    df["confidence"] = df["prediction_samples"].apply(calculate_confidence)

    # ---------------------------------------------------------
    # CALIBRATED PREDICTION
    # ---------------------------------------------------------

    df["raw_prediction_bonus"] = df["prediction_score"].apply(
        calibrate_prediction_score
    )

    df["prediction_impact"] = df.apply(
        lambda row: calculate_prediction_impact(
            row["prediction_score"],
            row["prediction_samples"],
        ),
        axis=1,
    )

    df["prediction_bonus"] = (
        df["raw_prediction_bonus"] * df["confidence"] * df["prediction_impact"]
    )

    # ---------------------------------------------------------
    # FINAL SCORE
    # ---------------------------------------------------------

    df["alpha_score"] = df["score"]

    df["final_score"] = df["alpha_score"] + df["prediction_bonus"]

    df["final_score"] = df["final_score"].clip(
        0,
        100,
    )

    df["score"] = df["final_score"]

    return df.sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)
