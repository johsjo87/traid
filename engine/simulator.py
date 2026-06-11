import pandas as pd
from engine.historical import load_price_history
from engine.features import build_feature_matrix
from engine.regime import detect_regime
from engine.alpha import build_alpha


def run_simulation(days: int = 30):

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "TSLA",
        "NFLX",
        "AMD",
        "AVGO",
        "JPM",
        "BAC",
        "GS",
        "MS",
        "LLY",
        "JNJ",
        "UNH",
        "ABBV",
        "KO",
        "MCD",
    ]

    prices = load_price_history(tickers)

    features = build_feature_matrix()

    equity = [1.0]

    # START POSITION (tom portfölj)
    position = {}

    max_days = min(len(prices), days)

    for day in range(1, max_days):
        # -----------------------------
        # SIGNALS
        # -----------------------------
        regime = detect_regime(features)
        alpha = build_alpha(features, regime)

        top = alpha.head(10).copy()

        total = top["score"].sum()

        new_position = {
            row["symbol"]: row["score"] / total for _, row in top.iterrows()
        }

        # -----------------------------
        # PnL (USE OLD POSITION)
        # -----------------------------
        daily_return = 0.0

        for sym, weight in position.items():
            if sym not in prices.columns:
                continue

            series = prices[sym]
            series = pd.Series(series).dropna().astype(float)

            if day >= len(series):
                continue

            prev_price = float(series.iloc[day - 1])
            curr_price = float(series.iloc[day])

            if prev_price == 0:
                continue

            r = (curr_price / prev_price) - 1
            daily_return += weight * r

        # -----------------------------
        # UPDATE EQUITY
        # -----------------------------
        equity.append(equity[-1] * (1 + daily_return))

        # -----------------------------
        # UPDATE POSITION
        # -----------------------------
        position = new_position

    return pd.Series(equity)
