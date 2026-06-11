import pandas as pd
import numpy as np


def build_portfolio_weights(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Converts alpha scores into normalized portfolio weights.
    """

    df = df.copy()

    # take top N
    top = df.sort_values("score", ascending=False).head(top_n).copy()

    # remove negative/zero scores
    top = top[top["score"] > 0]

    if top.empty:
        return pd.DataFrame()

    # convert score → weights
    top["weight"] = top["score"] / top["score"].sum()

    return top[["symbol", "score", "signal", "weight"]]
