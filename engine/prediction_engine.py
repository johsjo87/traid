import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_20",
    "rel_strength",
    "volatility",
    "return_5",
]


def calculate_reliability(samples, positive_rate):
    if samples < 20:
        return "LOW"

    if samples < 50:
        return "MEDIUM"

    if samples >= 100 and positive_rate >= 60:
        return "HIGH"

    return "MEDIUM"


def calculate_setup_quality(avg_return, positive_rate, samples):
    score = 50

    if avg_return > 0.05:
        score += 20
    elif avg_return > 0.02:
        score += 10
    elif avg_return < -0.05:
        score -= 20
    elif avg_return < -0.02:
        score -= 10

    if positive_rate > 70:
        score += 20
    elif positive_rate > 60:
        score += 10
    elif positive_rate < 30:
        score -= 20
    elif positive_rate < 40:
        score -= 10

    if samples > 100:
        score += 10
    elif samples < 20:
        score -= 5

    return max(0, min(100, score))


def calculate_prediction_score(
    avg_return,
    positive_rate,
    reliability,
):
    reliability_factor = {
        "HIGH": 1.0,
        "MEDIUM": 0.6,
        "LOW": 0.2,
    }[reliability]

    edge = avg_return * 100 + (positive_rate - 50) * 0.20

    score = edge * reliability_factor

    return round(float(score), 2)


def _calculate_similarity(current, history):
    available = [
        column
        for column in FEATURE_COLUMNS
        if column in current.index and column in history.columns
    ]

    if not available:
        return pd.Series(dtype=float, index=history.index)

    current_values = pd.to_numeric(
        current[available],
        errors="coerce",
    ).astype(float)

    history_values = (
        history[available]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype(float)
    )

    valid_history = history_values.notna().all(axis=1)

    history_values = history_values.loc[valid_history]

    if history_values.empty:
        return pd.Series(dtype=float, index=history.index)

    current_values = current_values.reindex(history_values.columns).astype(float)

    scales = history_values.std().replace(0, 1.0)

    normalized_current = current_values / scales
    normalized_history = history_values / scales

    differences = normalized_history - normalized_current

    distances = np.sqrt(
        np.asarray(
            (differences**2).sum(axis=1),
            dtype=float,
        )
    )

    return pd.Series(
        distances,
        index=history_values.index,
    )


def analyze_prediction_quality(
    current_features: pd.DataFrame,
    history: pd.DataFrame,
    sample_sizes=(10, 25, 50, 100),
):
    results = []

    if current_features is None or current_features.empty:
        return pd.DataFrame()

    if history is None or history.empty:
        return pd.DataFrame()

    if "future_return" not in history.columns:
        return pd.DataFrame()

    history = history.copy()

    history = history.dropna(
        subset=[column for column in FEATURE_COLUMNS if column in history.columns]
        + ["future_return"]
    )

    for _, current in current_features.iterrows():
        distances = _calculate_similarity(
            current,
            history,
        )

        if distances.empty:
            continue

        ordered = distances.sort_values()

        for sample_size in sample_sizes:
            if len(ordered) < sample_size:
                continue

            nearest_index = ordered.head(sample_size).index

            nearest_returns = pd.to_numeric(
                history.loc[
                    nearest_index,
                    "future_return",
                ],
                errors="coerce",
            ).dropna()

            if nearest_returns.empty:
                continue

            results.append(
                {
                    "symbol": current["symbol"],
                    "sample_size": sample_size,
                    "avg_future_return": nearest_returns.mean(),
                    "positive_rate": (nearest_returns > 0).mean() * 100,
                    "samples": len(nearest_returns),
                }
            )

    if not results:
        return pd.DataFrame()

    result = pd.DataFrame(results)

    return result


def predict_from_history(
    current_features: pd.DataFrame,
    history: pd.DataFrame,
    max_samples: int = 100,
):
    predictions = []

    if current_features is None or current_features.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "prediction_score",
                "setup_quality",
                "avg_future_return",
                "positive_rate",
                "reliability",
                "samples",
            ]
        )

    if history is None or history.empty:
        for _, current in current_features.iterrows():
            predictions.append(
                {
                    "symbol": current["symbol"],
                    "prediction_score": 0.0,
                    "setup_quality": 0,
                    "avg_future_return": None,
                    "positive_rate": None,
                    "reliability": "LOW",
                    "samples": 0,
                }
            )

        return pd.DataFrame(predictions)

    history = history.copy()

    if "future_return" not in history.columns:
        raise ValueError("History must contain future_return")

    history = history.dropna(
        subset=[column for column in FEATURE_COLUMNS if column in history.columns]
        + ["future_return"]
    )

    for _, current in current_features.iterrows():
        distances = _calculate_similarity(
            current,
            history,
        )

        if distances.empty:
            predictions.append(
                {
                    "symbol": current["symbol"],
                    "prediction_score": 0.0,
                    "setup_quality": 0,
                    "avg_future_return": None,
                    "positive_rate": None,
                    "reliability": "LOW",
                    "samples": 0,
                }
            )
            continue

        distances = distances.sort_values()

        nearest = distances.head(max_samples)

        similar = history.loc[nearest.index].copy()

        if similar.empty:
            predictions.append(
                {
                    "symbol": current["symbol"],
                    "prediction_score": 0.0,
                    "setup_quality": 0,
                    "avg_future_return": None,
                    "positive_rate": None,
                    "reliability": "LOW",
                    "samples": 0,
                }
            )
            continue

        distance_values = nearest.to_numpy(dtype=float)

        similarity_weights = 1.0 / (1.0 + distance_values)

        future_returns = pd.to_numeric(
            similar["future_return"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(future_returns) & np.isfinite(similarity_weights)

        future_returns = future_returns[valid]
        similarity_weights = similarity_weights[valid]

        samples = len(future_returns)

        if samples == 0:
            predictions.append(
                {
                    "symbol": current["symbol"],
                    "prediction_score": 0.0,
                    "setup_quality": 0,
                    "avg_future_return": None,
                    "positive_rate": None,
                    "reliability": "LOW",
                    "samples": 0,
                }
            )
            continue

        total_weight = similarity_weights.sum()

        if total_weight <= 0:
            avg_return = float(future_returns.mean())

            positive_rate = float((future_returns > 0).mean() * 100)

        else:
            avg_return = float(
                np.average(
                    future_returns,
                    weights=similarity_weights,
                )
            )

            positive_rate = float(
                np.average(
                    future_returns > 0,
                    weights=similarity_weights,
                )
                * 100
            )

        reliability = calculate_reliability(
            samples,
            positive_rate,
        )

        prediction_score = calculate_prediction_score(
            avg_return,
            positive_rate,
            reliability,
        )

        quality = calculate_setup_quality(
            avg_return,
            positive_rate,
            samples,
        )

        predictions.append(
            {
                "symbol": current["symbol"],
                "prediction_score": prediction_score,
                "setup_quality": quality,
                "avg_future_return": avg_return,
                "positive_rate": positive_rate,
                "reliability": reliability,
                "samples": samples,
            }
        )

    return pd.DataFrame(predictions)
