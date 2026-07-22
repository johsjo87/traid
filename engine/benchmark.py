import pandas as pd


def benchmark_stats(prices: pd.Series):

    returns = prices.pct_change().dropna()

    equity = (1 + returns).cumprod()

    peak = equity.cummax()

    drawdown = (peak - equity) / peak

    return {
        "equity": equity,
        "total_return": equity.iloc[-1] - 1,
        "max_drawdown": drawdown.max(),
    }
