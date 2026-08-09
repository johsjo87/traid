import pandas as pd

from engine.strategy import build_strategy
from engine.feature_dataset import build_feature_dataset
from engine.trade_engine import TradeManager


def run_simulation(
    prices: pd.DataFrame,
    window: int = 60,
    rebalance_freq: int = 5,
    use_prediction: bool = True,
):
    """
    Backtestar TRAID över historisk data.

    Strategin byggs om endast vid rebalance.

    Research-datasetet byggs en gång per simulation.
    Vid varje rebalance får strategin endast använda
    research-observationer vars decision_date ligger
    före det aktuella beslutstillfället.

    Detta säkerställer att prediction engine inte får
    tillgång till framtida research-observationer.

    Decision log sparas för att kunna analysera exakt
    hur prediction påverkar portföljbesluten.
    """

    if len(prices) < window + 20:
        raise ValueError("Not enough data")

    # --------------------------------------------------
    # BUILD RESEARCH DATASET ONCE
    # --------------------------------------------------

    research_dataset = build_feature_dataset(prices)

    if research_dataset.empty:
        raise ValueError("Research dataset is empty")

    # Ensure decision_date is datetime.
    research_dataset["decision_date"] = pd.to_datetime(
        research_dataset["decision_date"]
    )

    # --------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------

    equity = [1.0]

    position = {}

    trade_manager = TradeManager()

    peak_equity = 1.0
    max_drawdown = 0.0

    # --------------------------------------------------
    # DECISION LOG
    # --------------------------------------------------

    decision_log = []

    start = window + 50

    if len(prices) <= start:
        raise ValueError("Not enough data")

    # --------------------------------------------------
    # SIMULATION
    # --------------------------------------------------

    for day in range(start, len(prices)):
        # --------------------------------------------------
        # REBALANCE
        # --------------------------------------------------

        if day % rebalance_freq == 0:
            # Everything before the current day is known.
            window_prices = prices.iloc[:day]

            current_date = pd.Timestamp(prices.index[day])

            # --------------------------------------------------
            # HISTORICAL RESEARCH DATA ONLY
            # --------------------------------------------------

            available_dataset = research_dataset[
                research_dataset["decision_date"] < current_date
            ].copy()

            # No future fallback is allowed.
            if available_dataset.empty:
                position = {}
                continue

            # --------------------------------------------------
            # BUILD STRATEGY
            # --------------------------------------------------

            result = build_strategy(
                window_prices,
                available_dataset,
                top_n=10,
                use_prediction=use_prediction,
            )

            portfolio = result["portfolio"]

            # --------------------------------------------------
            # SAVE DECISION
            # --------------------------------------------------

            if portfolio is not None and not portfolio.empty:
                decision_log.append(
                    {
                        "date": current_date,
                        "use_prediction": use_prediction,
                        "symbols": list(portfolio["symbol"]),
                        "scores": {
                            row["symbol"]: float(row["score"])
                            for _, row in portfolio.iterrows()
                        },
                        "weights": {
                            row["symbol"]: float(row["weight"])
                            for _, row in portfolio.iterrows()
                        },
                    }
                )

            else:
                decision_log.append(
                    {
                        "date": current_date,
                        "use_prediction": use_prediction,
                        "symbols": [],
                        "scores": {},
                        "weights": {},
                    }
                )

            # --------------------------------------------------
            # CURRENT PRICES
            # --------------------------------------------------

            current_prices = {}

            for symbol in prices.columns:
                series = prices[symbol].dropna()

                if day < len(series):
                    current_prices[symbol] = float(series.iloc[day])

            # --------------------------------------------------
            # DESIRED POSITIONS
            # --------------------------------------------------

            if portfolio is not None and not portfolio.empty:
                desired_positions = set(portfolio["symbol"])
            else:
                desired_positions = set()

            existing_positions = set(trade_manager.open_trades.keys())

            # --------------------------------------------------
            # CLOSE REMOVED POSITIONS
            # --------------------------------------------------

            for symbol in existing_positions - desired_positions:
                if symbol in current_prices:
                    trade_manager.close_trade(
                        symbol=symbol,
                        date=prices.index[day],
                        price=current_prices[symbol],
                        reason="Rebalance",
                    )

            # --------------------------------------------------
            # OPEN NEW POSITIONS
            # --------------------------------------------------

            for symbol in desired_positions - existing_positions:
                if symbol in current_prices:
                    trade_manager.open_trade(
                        symbol=symbol,
                        date=prices.index[day],
                        price=current_prices[symbol],
                    )

            # --------------------------------------------------
            # UPDATE WEIGHTS
            # --------------------------------------------------

            if portfolio is None or portfolio.empty:
                position = {}

            else:
                total_weight = portfolio["weight"].sum()

                if total_weight <= 0:
                    position = {}

                else:
                    position = {
                        row["symbol"]: row["weight"] / total_weight
                        for _, row in portfolio.iterrows()
                    }

        # --------------------------------------------------
        # DAILY RETURN
        # --------------------------------------------------

        daily_return = 0.0

        for symbol, weight in position.items():
            if symbol not in prices.columns:
                continue

            series = prices[symbol].dropna()

            if day >= len(series):
                continue

            if day == 0:
                continue

            prev_price = float(series.iloc[day - 1])
            curr_price = float(series.iloc[day])

            if prev_price <= 0:
                continue

            symbol_return = (curr_price / prev_price) - 1

            daily_return += weight * symbol_return

        # --------------------------------------------------
        # UPDATE EQUITY
        # --------------------------------------------------

        new_equity = equity[-1] * (1 + daily_return)

        equity.append(new_equity)

        # --------------------------------------------------
        # DRAWDOWN
        # --------------------------------------------------

        if new_equity > peak_equity:
            peak_equity = new_equity

        drawdown = (peak_equity - new_equity) / peak_equity

        max_drawdown = max(
            max_drawdown,
            drawdown,
        )

    # --------------------------------------------------
    # CLOSE REMAINING TRADES
    # --------------------------------------------------

    final_prices = {}

    for symbol in prices.columns:
        series = prices[symbol].dropna()

        if not series.empty:
            final_prices[symbol] = float(series.iloc[-1])

    trade_manager.close_all(
        date=prices.index[-1],
        prices=final_prices,
        reason="End of backtest",
    )

    # --------------------------------------------------
    # TRADE STATISTICS
    # --------------------------------------------------

    trade_stats = trade_manager.statistics()

    equity = pd.Series(equity)

    # --------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------

    stats = {
        "total_return": equity.iloc[-1] - 1,
        "trades": trade_stats["total_trades"],
        "win_rate": trade_stats["win_rate"],
        "average_gain": trade_stats["average_win"],
        "average_loss": trade_stats["average_loss"],
        "profit_factor": trade_stats["profit_factor"],
        "max_drawdown": max_drawdown,
        "closed_trades": trade_manager.get_closed_trades(),
        # --------------------------------------------------
        # DIAGNOSTIC DATA
        # --------------------------------------------------
        "decision_log": decision_log,
    }

    return equity, stats
