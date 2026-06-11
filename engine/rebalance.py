import pandas as pd
import numpy as np


def simulate_rebalance(portfolio_snapshots: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Simulates portfolio over time.
    Each snapshot = new rebalance period.
    """

    equity = 1.0
    equity_curve = []

    for snapshot in portfolio_snapshots:
        if snapshot is None or snapshot.empty:
            continue

        # simple return model (placeholder but consistent)
        period_return = (snapshot["score"] / 100).mean() * 0.02  # 2% scaling factor

        equity *= 1 + period_return

        equity_curve.append(equity)

    return pd.DataFrame(
        {
            "equity": equity_curve,
            "return": pd.Series(equity_curve).pct_change().fillna(0),
        }
    )
