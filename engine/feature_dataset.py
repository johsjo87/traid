import pandas as pd

from engine.features import build_feature_matrix


def build_feature_dataset(
    prices: pd.DataFrame,
    lookback: int = 60,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Bygger ett historiskt research-dataset.

    Varje rad representerar ett historiskt beslutstillfälle och innehåller:

    - decision_date
    - symbol
    - features som var tillgängliga vid beslutet
    - future_return som endast används som target/resultat

    Viktigt:
    Signalfeatures får endast använda data fram till decision_date.
    Future return beräknas först efter decision_date.
    """

    rows = []

    if prices.empty:
        return pd.DataFrame()

    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    if lookback <= 0:
        raise ValueError("lookback must be greater than 0")

    # ---------------------------------------------------------
    # HISTORICAL DECISION DATES
    # ---------------------------------------------------------

    for day in range(lookback, len(prices)):
        # All information available at the decision point.
        history = prices.iloc[:day]

        if history.empty:
            continue

        decision_date = history.index[-1]

        features = build_feature_matrix(history)

        if features.empty:
            continue

        # -----------------------------------------------------
        # EACH SYMBOL
        # -----------------------------------------------------

        for _, feature in features.iterrows():
            symbol = feature["symbol"]

            if symbol not in prices.columns:
                continue

            # -------------------------------------------------
            # SYMBOL-SPECIFIC VALID PRICE HISTORY
            # -------------------------------------------------

            symbol_prices = prices[symbol].dropna()

            if symbol_prices.empty:
                continue

            # Last price that was actually available
            # at the decision date.
            historical_prices = symbol_prices.loc[symbol_prices.index <= decision_date]

            if len(historical_prices) == 0:
                continue

            decision_position = len(historical_prices) - 1

            # We need `horizon` future observations.
            future_position = decision_position + horizon

            if future_position >= len(symbol_prices):
                continue

            p0 = float(historical_prices.iloc[-1])
            p1 = float(symbol_prices.iloc[future_position])

            if p0 <= 0:
                continue

            future_return = (p1 / p0) - 1

            # -------------------------------------------------
            # STORE RESEARCH OBSERVATION
            # -------------------------------------------------

            row = feature.to_dict()

            row["decision_date"] = decision_date
            row["future_return"] = future_return

            rows.append(row)

    dataset = pd.DataFrame(rows)

    if dataset.empty:
        return dataset

    # Keep dates explicit and sorted.
    dataset["decision_date"] = pd.to_datetime(dataset["decision_date"])

    dataset = dataset.sort_values(["decision_date", "symbol"]).reset_index(drop=True)

    return dataset
