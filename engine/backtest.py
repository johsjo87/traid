import pandas as pd
import yfinance as yf
import numpy as np


def backtest_signals(signal_df: pd.DataFrame, lookback="1y"):

    equity_curves = {}

    for _, row in signal_df.iterrows():
        symbol = row["symbol"]
        signal = row["signal"]

        df = yf.download(symbol, period=lookback, interval="1d", progress=False)

        if df.empty:
            continue

        df = df.copy()
        df["ret"] = df["Close"].pct_change().fillna(0)

        # position
        if signal == "BUY":
            df["position"] = 1
        elif signal == "SELL":
            df["position"] = -1
        else:
            df["position"] = 0

        df["strategy_ret"] = df["position"].shift(1) * df["ret"]

        df["equity"] = (1 + df["strategy_ret"]).cumprod()

        equity_curves[symbol] = df["equity"]

    eq = pd.DataFrame(equity_curves)

    eq = eq.ffill().fillna(1.0)

    eq["portfolio"] = eq.mean(axis=1)

    # metrics
    returns = eq["portfolio"].pct_change().fillna(0)

    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    total_return = eq["portfolio"].iloc[-1] - 1
    drawdown = (eq["portfolio"] / eq["portfolio"].cummax() - 1).min()

    summary = pd.DataFrame(
        [{"total_return": total_return, "sharpe": sharpe, "max_drawdown": drawdown}]
    )

    return summary
