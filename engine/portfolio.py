import pandas as pd


def build_portfolio_weights(
    df: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    """
    Bygger portfölj beroende på regim.
    Vikter baseras på score-styrka.
    """

    df = df.copy()

    # -------------------------
    # PORTFOLIO SIZE
    # -------------------------

    if regime == "BULL":
        top_n = 15

    elif regime == "NEUTRAL":
        top_n = 10

    else:
        top_n = 5

    # -------------------------
    # SELECT HOLDINGS
    # -------------------------

    top = df.sort_values("score", ascending=False).head(top_n).copy()

    top = top[top["score"] > 0]

    if top.empty:
        return pd.DataFrame()

    # -------------------------
    # WEIGHTS
    # -------------------------

    top["weight"] = top["score"] / top["score"].sum()

    # -------------------------
    # OUTPUT
    # -------------------------

    columns = [
        "symbol",
        "alpha_score",
        "prediction_bonus",
        "confidence",
        "prediction_samples",
        "score",
        "signal",
        "weight",
    ]

    columns = [c for c in columns if c in top.columns]

    return top[columns]
