import pandas as pd
from engine.strategy import build_strategy


def run_simulation(prices: pd.DataFrame, window: int = 60, rebalance_freq: int = 5):

    equity = [1.0]
    position = {}

    start = window + 50

    if len(prices) < start:
        raise ValueError("Not enough data")

    for day in range(start, len(prices)):
        # rebalancera bara ibland (inte varje dag)
        if day % rebalance_freq == 0:
            window_prices = prices.iloc[:day]

            result = build_strategy(window_prices, top_n=10)
            portfolio = result["portfolio"]

            if portfolio is None or portfolio.empty:
                position = {}
            else:
                total = portfolio["weight"].sum()

                position = {
                    row["symbol"]: row["weight"] / total
                    for _, row in portfolio.iterrows()
                }

        # PnL (mark-to-market)
        daily_return = 0.0

        for sym, weight in position.items():
            if sym not in prices.columns:
                continue

            series = prices[sym].dropna()

            if day >= len(series):
                continue

            prev_price = float(series.iloc[day - 1])
            curr_price = float(series.iloc[day])

            if prev_price == 0:
                continue

            daily_return += weight * ((curr_price / prev_price) - 1)

        equity.append(equity[-1] * (1 + daily_return))

    return pd.Series(equity)
