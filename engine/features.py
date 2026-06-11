import pandas as pd


def build_feature_matrix(prices: pd.DataFrame):

    rows = []

    for symbol in prices.columns:
        close = prices[symbol].dropna()

        if len(close) < 60:
            continue

        price = float(close.iloc[-1])

        return_5 = float(close.pct_change(5).iloc[-1])
        return_20 = float(close.pct_change(20).iloc[-1])

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

        trend_strength = float(sma20 / sma50)

        volatility = float(close.pct_change().rolling(20).std().iloc[-1])

        rolling_high = close.rolling(126).max().iloc[-1]
        distance_high = float(price / rolling_high)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()

        rs = gain / (loss + 1e-9)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])

        rows.append(
            {
                "symbol": symbol,
                "price": price,
                "return_5": return_5,
                "return_20": return_20,
                "trend_strength": trend_strength,
                "volatility": volatility,
                "distance_high": distance_high,
                "rsi": rsi,
            }
        )

    return pd.DataFrame(rows)
