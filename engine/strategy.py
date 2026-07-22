import pandas as pd

from engine.features import build_feature_matrix
from engine.regime import detect_regime
from engine.alpha import build_alpha
from engine.portfolio import build_portfolio_weights
from engine.prediction_engine import predict_from_history
from engine.scoring import combine_scores


def build_strategy(
    prices: pd.DataFrame,
    dataset: pd.DataFrame,
    top_n: int = 10,
):

    features = build_feature_matrix(prices)

    if features is None or len(features) == 0:
        raise ValueError("Feature matrix is empty")

    regime = detect_regime(features)

    alpha = build_alpha(features, regime)
    predictions = predict_from_history(
        features,
        dataset,
    )

    if alpha is None or len(alpha) == 0:
        raise ValueError("Alpha is empty")
    alpha = combine_scores(
        alpha,
        predictions,
    )

    portfolio = build_portfolio_weights(
        alpha,
        regime,
    )

    return {
        "features": features,
        "regime": regime,
        "alpha": alpha,
        "predictions": predictions,
        "portfolio": portfolio,
    }
