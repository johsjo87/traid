import pandas as pd

from engine.features import build_feature_matrix
from engine.regime import detect_regime
from engine.alpha import build_alpha
from engine.portfolio import build_portfolio_weights


def build_strategy(prices: pd.DataFrame, top_n: int = 10):

    features = build_feature_matrix(prices)

    if features is None or len(features) == 0:
        raise ValueError("Feature matrix is empty")

    regime = detect_regime(features)

    alpha = build_alpha(features, regime)

    if alpha is None or len(alpha) == 0:
        raise ValueError("Alpha is empty")

    portfolio = build_portfolio_weights(alpha, top_n=top_n)

    return {
        "features": features,
        "regime": regime,
        "alpha": alpha,
        "portfolio": portfolio,
    }
