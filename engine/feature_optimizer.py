import pandas as pd


def optimize_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Testar olika feature-viktningar
    mot historiskt framtida utfall.
    """

    if dataset.empty:
        return pd.DataFrame()

    weights = [
        [0.4, 0.3, 0.2, 0.1],
        [0.3, 0.4, 0.2, 0.1],
        [0.4, 0.2, 0.3, 0.1],
        [0.3, 0.3, 0.3, 0.1],
        [0.5, 0.3, 0.1, 0.1],
    ]

    results = []

    for w in weights:
        score = (
            dataset["return_20"] * w[0]
            + dataset["rel_strength"] * w[1]
            + dataset["return_5"] * w[2]
            - dataset["volatility"] * w[3]
        )

        correlation = score.corr(dataset["future_return"])

        results.append(
            {
                "return_20_weight": w[0],
                "rel_strength_weight": w[1],
                "return_5_weight": w[2],
                "volatility_weight": w[3],
                "correlation": correlation,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )
