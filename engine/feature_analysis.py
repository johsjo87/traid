import pandas as pd


def analyze_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Analyserar vilka features som historiskt
    haft samband med framtida avkastning.
    """

    if dataset.empty:
        return pd.DataFrame()

    features = [
        "return_5",
        "return_20",
        "trend_strength",
        "volatility",
        "distance_high",
        "rsi",
        "rel_strength",
    ]

    rows = []

    for feature in features:
        if feature not in dataset.columns:
            continue

        correlation = dataset[feature].corr(dataset["future_return"])

        rows.append(
            {
                "feature": feature,
                "correlation": correlation,
                "mean": dataset[feature].mean(),
                "std": dataset[feature].std(),
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values("correlation", ascending=False).reset_index(drop=True)
