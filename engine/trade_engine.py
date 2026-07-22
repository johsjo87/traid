from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    symbol: str

    entry_date: object
    entry_price: float

    exit_date: Optional[object] = None
    exit_price: Optional[float] = None

    exit_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def holding_days(self) -> Optional[int]:
        if self.is_open:
            return None

        try:
            return (self.exit_date - self.entry_date).days
        except Exception:
            return None

    @property
    def return_pct(self) -> Optional[float]:
        if self.is_open:
            return None

        return (self.exit_price / self.entry_price) - 1

    @property
    def pnl_pct(self) -> Optional[float]:
        return self.return_pct

    @property
    def winner(self) -> Optional[bool]:
        if self.is_open:
            return None

        return self.return_pct > 0


class TradeManager:
    def __init__(self):
        self.open_trades: dict[str, Trade] = {}
        self.closed_trades: list[Trade] = []

    def open_trade(
        self,
        symbol,
        date,
        price,
    ):
        if symbol in self.open_trades:
            return

        self.open_trades[symbol] = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=price,
        )

    def close_trade(
        self,
        symbol,
        date,
        price,
        reason="Signal",
    ):
        trade = self.open_trades.pop(symbol, None)

        if trade is None:
            return

        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason

        self.closed_trades.append(trade)

    def close_all(
        self,
        date,
        prices: dict,
        reason="Rebalance",
    ):
        for symbol in list(self.open_trades.keys()):
            if symbol not in prices:
                continue

            self.close_trade(
                symbol=symbol,
                date=date,
                price=prices[symbol],
                reason=reason,
            )

    def get_closed_trades(self):
        return self.closed_trades

    def statistics(self) -> dict:

        if not self.closed_trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "average_return": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "profit_factor": 0.0,
            }

        returns = [t.return_pct for t in self.closed_trades]

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        total_profit = sum(wins)
        total_loss = abs(sum(losses))

        return {
            "total_trades": len(returns),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(returns),
            "average_return": sum(returns) / len(returns),
            "average_win": sum(wins) / len(wins) if wins else 0.0,
            "average_loss": sum(losses) / len(losses) if losses else 0.0,
            "profit_factor": (
                total_profit / total_loss if total_loss > 0 else float("inf")
            ),
        }
