import pandas as pd


def analyze_features(features: pd.DataFrame) -> pd.DataFrame:

    if features is None or features.empty:
        return pd.DataFrame()

    metrics = [
        "return_5",
        "return_20",
        "trend_strength",
        "volatility",
        "distance_high",
        "rsi",
        "rel_strength",
    ]

    rows = []

    for metric in metrics:
        if metric not in features.columns:
            continue

        series = features[metric]

        rows.append(
            {
                "feature": metric,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    return pd.DataFrame(rows)
