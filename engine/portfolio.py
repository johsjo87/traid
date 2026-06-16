import pandas as pd


def build_portfolio_weights(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:

    df = df.copy()

    top = df.sort_values("score", ascending=False).head(top_n).copy()
    top = top[top["score"] > 0]

    if top.empty:
        return pd.DataFrame()

    top["weight"] = top["score"] / top["score"].sum()

    return top[["symbol", "score", "signal", "weight"]]
