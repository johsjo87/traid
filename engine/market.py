from __future__ import annotations

import yfinance as yf
import pandas as pd

from config import TICKERS, LOOKBACK_PERIOD


def _latest_close(data: pd.DataFrame) -> float:
    """
    Returnerar senaste stängningskursen oavsett om
    yfinance returnerar vanlig DataFrame eller MultiIndex.
    """

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return float(close.iloc[-1])


def load_market_snapshot() -> pd.DataFrame:
    """
    Hämtar senaste marknadsinformationen för samtliga tickers.

    Returnerar:

    symbol | price
    ----------------
    AAPL   | 203.55
    MSFT   | 518.10
    ...
    """

    rows = []

    for symbol in TICKERS:
        try:
            df = yf.download(
                symbol,
                period=LOOKBACK_PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "price": _latest_close(df),
                }
            )

        except Exception as e:
            print(f"[WARNING] {symbol}: {e}")

    market = pd.DataFrame(rows)

    market = market.sort_values(by="symbol").reset_index(drop=True)

    return market
