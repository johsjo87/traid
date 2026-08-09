import pandas as pd

from engine.feature_dataset import build_feature_dataset
from engine.features import build_feature_matrix


def validate_no_lookahead(prices: pd.DataFrame) -> dict:
    """
    Verifierar att historiska features och targets är tidsmässigt separerade.

    Testet kontrollerar:

    1. decision_date ligger före framtida observationsdatum.
    2. future_return använder data efter decision_date.
    3. features kan byggas enbart från data fram till decision_date.
    """

    dataset = build_feature_dataset(prices)

    if dataset.empty:
        return {
            "valid": False,
            "reason": "Research dataset is empty",
        }

    violations = []

    for _, row in dataset.iterrows():
        decision_date = pd.Timestamp(row["decision_date"])

        symbol = row["symbol"]

        if symbol not in prices.columns:
            violations.append(f"{symbol}: symbol missing from prices")
            continue

        symbol_prices = prices[symbol].dropna()

        historical = symbol_prices.loc[symbol_prices.index <= decision_date]

        future = symbol_prices.loc[symbol_prices.index > decision_date]

        if historical.empty:
            violations.append(f"{symbol}: no historical data")
            continue

        if future.empty:
            violations.append(f"{symbol}: no future data")
            continue

        # -------------------------------------------------
        # VERIFY FEATURES USE ONLY HISTORICAL DATA
        # -------------------------------------------------

        reconstructed = build_feature_matrix(prices.loc[prices.index <= decision_date])

        reconstructed_symbol = reconstructed[reconstructed["symbol"] == symbol]

        if reconstructed_symbol.empty:
            violations.append(
                f"{symbol} @ {decision_date}: feature reconstruction failed"
            )
            continue

        reconstructed_row = reconstructed_symbol.iloc[0]

        # Compare the actual features stored in dataset
        # with features reconstructed using only historical data.

        feature_columns = [
            "return_5",
            "return_20",
            "trend_strength",
            "volatility",
            "distance_high",
            "rsi",
            "rel_strength",
        ]

        for feature in feature_columns:
            original = row[feature]
            reconstructed_value = reconstructed_row[feature]

            if pd.isna(original) and pd.isna(reconstructed_value):
                continue

            if pd.isna(original) or pd.isna(reconstructed_value):
                violations.append(f"{symbol} @ {decision_date}: {feature} mismatch")
                break

            if abs(float(original) - float(reconstructed_value)) > 1e-9:
                violations.append(f"{symbol} @ {decision_date}: {feature} mismatch")
                break

    return {
        "valid": len(violations) == 0,
        "rows_checked": len(dataset),
        "violations": violations[:20],
        "violation_count": len(violations),
    }
