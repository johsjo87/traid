import pandas as pd


def build_portfolio_weights(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:

    df = df.copy()

    top = df.sort_values("score", ascending=False).head(top_n).copy()

    top = top[top["score"] > 0]

    if top.empty:
        return pd.DataFrame()

    top["weight"] = top["score"] / top["score"].sum()

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

    # Säkerhet om någon kolumn saknas
    columns = [col for col in columns if col in top.columns]

    return top[columns]
