import pandas as pd


def build_feature_matrix(prices: pd.DataFrame):

    rows = []

    market_return = prices.pct_change().mean(axis=1)

    for symbol in prices.columns:
        close = prices[symbol].dropna()

        if len(close) < 30:
            continue

        price = float(close.iloc[-1])

        return_5 = float(close.pct_change(5).iloc[-1])
        return_20 = float(close.pct_change(20).iloc[-1])

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

        trend_strength = (
            float(sma20 / sma50)
            if sma20 == sma20 and sma50 == sma50 and sma50 != 0
            else 1.0
        )

        volatility = float(close.pct_change().rolling(20).std().iloc[-1])
        if pd.isna(volatility):
            volatility = 0.0

        rolling_high = close.rolling(126).max().iloc[-1]
        distance_high = (
            float(price / rolling_high)
            if rolling_high == rolling_high and rolling_high != 0
            else 1.0
        )

        # RSI (robust fallback)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()

        rs = gain / (loss + 1e-9)
        rsi_series = (100 - (100 / (1 + rs))).iloc[-1]

        rsi = float(rsi_series) if rsi_series == rsi_series else 50.0

        # RELATIVE STRENGTH (robust)
        stock_ret_20 = close.pct_change(20).iloc[-1]
        market_ret_20 = market_return.iloc[-1]

        if pd.isna(stock_ret_20) or pd.isna(market_ret_20):
            rel_strength = 0.0
        else:
            rel_strength = float(stock_ret_20 - market_ret_20)

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
                "rel_strength": rel_strength,
            }
        )

    df = pd.DataFrame(rows)

    return df
