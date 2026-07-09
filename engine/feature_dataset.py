import pandas as pd

from engine.features import build_feature_matrix


def build_feature_dataset(
    prices: pd.DataFrame,
    lookback: int = 60,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Bygger ett dataset för forskning.

    Varje rad innehåller:
    - features vid en viss dag
    - den faktiska framtida avkastningen
    """

    rows = []

    last_day = len(prices) - horizon

    for day in range(lookback, last_day):
        history = prices.iloc[:day]

        features = build_feature_matrix(history)

        if features.empty:
            continue

        for _, feature in features.iterrows():
            symbol = feature["symbol"]

            if symbol not in prices.columns:
                continue

            future_prices = prices[symbol].dropna()

            if day + horizon >= len(future_prices):
                continue

            p0 = float(future_prices.iloc[day])
            p1 = float(future_prices.iloc[day + horizon])

            future_return = (p1 / p0) - 1

            row = feature.to_dict()
            row["future_return"] = future_return

            rows.append(row)

    return pd.DataFrame(rows)
