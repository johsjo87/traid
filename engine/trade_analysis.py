import pandas as pd


def analyze_trades(trades: list[dict]) -> dict:
    """
    Analyserar alla avslutade affärer.
    """

    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "best_trade": None,
            "worst_trade": None,
            "average_holding_days": 0.0,
            "expectancy": 0.0,
        }

    df = pd.DataFrame(trades)

    # -------------------------
    # RETURNS
    # -------------------------

    df["return"] = (df["exit_price"] / df["entry_price"]) - 1

    # -------------------------
    # HOLDING PERIOD
    # -------------------------

    df["holding_days"] = (
        pd.to_datetime(df["exit_date"]) - pd.to_datetime(df["entry_date"])
    ).dt.days

    # -------------------------
    # WINNERS / LOSERS
    # -------------------------

    winners = df[df["return"] > 0]
    losers = df[df["return"] <= 0]

    win_rate = len(winners) / len(df)

    avg_winner = winners["return"].mean() if not winners.empty else 0.0
    avg_loser = losers["return"].mean() if not losers.empty else 0.0

    expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)

    # -------------------------
    # BEST / WORST
    # -------------------------

    best_trade = (
        winners.sort_values("return", ascending=False).iloc[0].to_dict()
        if not winners.empty
        else None
    )

    worst_trade = (
        losers.sort_values("return").iloc[0].to_dict() if not losers.empty else None
    )

    return {
        "total_trades": len(df),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": win_rate,
        "average_return": df["return"].mean(),
        "average_winner": avg_winner,
        "average_loser": avg_loser,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "average_holding_days": df["holding_days"].mean(),
        "expectancy": expectancy,
    }
