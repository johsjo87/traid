import yfinance as yf
import pandas as pd
from typing import Optional

_price_cache: Optional[pd.DataFrame] = None


def load_price_history(tickers, period="6mo"):

    global _price_cache

    if _price_cache is not None:
        return _price_cache

    data = {}

    for t in tickers:
        try:
            df = yf.download(
                t,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if df is None or df.empty:
                print(f"NO DATA: {t}")
                continue

            close = df["Close"]

            # Säkerställ att Close alltid blir en 1D Series
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna().astype(float)

            if len(close) < 10:
                print(f"TOO LITTLE DATA: {t}")
                continue

            close.name = t

            data[t] = close

        except Exception as e:
            print(f"ERROR {t}: {e}")
            continue

    if len(data) == 0:
        raise ValueError("No valid price data loaded")

    prices = pd.DataFrame(data)

    _price_cache = prices

    print(f"CACHED PRICES: {len(prices.columns)} symbols")

    return prices
