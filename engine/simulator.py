import pandas as pd

from engine.strategy import build_strategy
from engine.feature_dataset import build_feature_dataset
from engine.trade_engine import TradeManager


def run_simulation(
    prices: pd.DataFrame,
    window: int = 60,
    rebalance_freq: int = 5,
):

    equity = [1.0]
    position = {}

    trade_manager = TradeManager()

    trade_returns = []

    peak_equity = 1.0
    max_drawdown = 0.0

    start = window + 50

    if len(prices) < start:
        raise ValueError("Not enough data")

    for day in range(start, len(prices)):
        # -------------------------
        # REBALANCE
        # -------------------------

        if day % rebalance_freq == 0:
            window_prices = prices.iloc[:day]

            research_dataset = build_feature_dataset(window_prices)

            if research_dataset.empty:
                continue

            result = build_strategy(
                window_prices,
                research_dataset,
                top_n=10,
            )

            portfolio = result["portfolio"]

            current_prices = {}

            for symbol in prices.columns:
                series = prices[symbol].dropna()

                if day < len(series):
                    current_prices[symbol] = float(series.iloc[day])

            desired_positions = (
                set(portfolio["symbol"])
                if portfolio is not None and not portfolio.empty
                else set()
            )

            existing_positions = set(trade_manager.open_trades.keys())

            # -------------------------
            # CLOSE REMOVED POSITIONS
            # -------------------------

            for symbol in existing_positions - desired_positions:
                if symbol in current_prices:
                    trade_manager.close_trade(
                        symbol=symbol,
                        date=prices.index[day],
                        price=current_prices[symbol],
                        reason="Rebalance",
                    )

            # -------------------------
            # OPEN NEW POSITIONS
            # -------------------------

            for symbol in desired_positions - existing_positions:
                if symbol in current_prices:
                    trade_manager.open_trade(
                        symbol=symbol,
                        date=prices.index[day],
                        price=current_prices[symbol],
                    )

            # -------------------------
            # UPDATE PORTFOLIO WEIGHTS
            # -------------------------

            if portfolio is None or portfolio.empty:
                position = {}

            else:
                total = portfolio["weight"].sum()

                position = {
                    row["symbol"]: row["weight"] / total
                    for _, row in portfolio.iterrows()
                }

        # -------------------------
        # DAILY RETURN
        # -------------------------

        daily_return = 0.0

        for symbol, weight in position.items():
            if symbol not in prices.columns:
                continue

            series = prices[symbol].dropna()

            if day >= len(series):
                continue

            prev_price = float(series.iloc[day - 1])
            curr_price = float(series.iloc[day])

            if prev_price == 0:
                continue

            daily_return += weight * ((curr_price / prev_price) - 1)

        equity.append(equity[-1] * (1 + daily_return))

        trade_returns.append(daily_return)

        if equity[-1] > peak_equity:
            peak_equity = equity[-1]

        drawdown = (peak_equity - equity[-1]) / peak_equity

        max_drawdown = max(max_drawdown, drawdown)

    # -------------------------
    # CLOSE REMAINING TRADES
    # -------------------------

    final_prices = {
        symbol: float(prices[symbol].dropna().iloc[-1]) for symbol in prices.columns
    }

    trade_manager.close_all(
        date=prices.index[-1],
        prices=final_prices,
        reason="End of backtest",
    )

    closed_trades = trade_manager.get_closed_trades()

    # -------------------------
    # PERFORMANCE
    # -------------------------

    equity = pd.Series(equity)

    trade_manager.close_all(
        date=prices.index[-1],
        prices={symbol: float(prices[symbol].iloc[-1]) for symbol in prices.columns},
        reason="End of backtest",
    )

    trade_stats = trade_manager.statistics()

    stats = {
        "total_return": equity.iloc[-1] - 1,
        "trades": trade_stats["total_trades"],
        "win_rate": trade_stats["win_rate"],
        "average_gain": trade_stats["average_win"],
        "average_loss": trade_stats["average_loss"],
        "profit_factor": trade_stats["profit_factor"],
        "max_drawdown": max_drawdown,
        "closed_trades": closed_trades,
    }

    return equity, stats
