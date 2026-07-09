import pandas as pd


def build_feature_matrix(prices: pd.DataFrame):

    rows = []

    market_return = prices.pct_change().mean(axis=1)

    for symbol in prices.columns:
        close = prices[symbol].dropna()

        if len(close) < 30:
            continue

        price = float(close.iloc[-1])

        # -------------------------
        # MOMENTUM
        # -------------------------

        return_5 = float(close.pct_change(5).iloc[-1])
        return_20 = float(close.pct_change(20).iloc[-1])

        # -------------------------
        # TREND
        # -------------------------

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

        if pd.isna(sma20) or pd.isna(sma50) or sma50 == 0:
            trend_strength = 1.0
        else:
            trend_strength = float(sma20 / sma50)

        # -------------------------
        # VOLATILITY
        # -------------------------

        volatility = float(close.pct_change().rolling(20).std().iloc[-1])

        if pd.isna(volatility):
            volatility = 0.0

        # -------------------------
        # DISTANCE TO HIGH
        # -------------------------

        lookback_high = min(126, len(close))

        rolling_high = close.rolling(lookback_high).max().iloc[-1]

        if rolling_high > 0:
            distance_high = float(price / rolling_high)
        else:
            distance_high = 1.0

        # -------------------------
        # RSI
        # -------------------------

        delta = close.diff()

        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()

        rs = gain / (loss + 1e-9)

        rsi_value = (100 - (100 / (1 + rs))).iloc[-1]

        if pd.isna(rsi_value):
            rsi = 50.0
        else:
            rsi = float(rsi_value)

        # -------------------------
        # RELATIVE STRENGTH
        # -------------------------

        stock_ret_20 = close.pct_change(20).iloc[-1]
        market_ret_20 = market_return.iloc[-1]

        if pd.isna(stock_ret_20) or pd.isna(market_ret_20):
            rel_strength = 0.0
        else:
            rel_strength = float(stock_ret_20 - market_ret_20)

        # -------------------------
        # STORE FEATURES
        # -------------------------

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

    return pd.DataFrame(rows)
