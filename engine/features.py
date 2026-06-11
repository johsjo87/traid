import yfinance as yf
import pandas as pd

TICKERS = [
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
    "ORCL",
    "CRM",
    "ADBE",
    "INTC",
    "QCOM",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "V",
    "MA",
    "LLY",
    "JNJ",
    "UNH",
    "ABBV",
    "PFE",
    "COST",
    "WMT",
    "KO",
    "PEP",
    "MCD",
    "CAT",
    "GE",
    "HON",
    "DE",
    "XOM",
    "CVX",
    "COP",
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
]


def build_feature_matrix():

    rows = []

    for symbol in TICKERS:
        try:
            df = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,  # <- viktig ändring
            )

            if df is None or df.empty:
                print(f"NO DATA: {symbol}")
                continue

            close = df["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.dropna()

            if len(close) < 60:
                print(f"TOO LITTLE DATA: {symbol}")
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

        except Exception as e:
            print(f"ERROR {symbol}: {e}")
            continue

    print(f"SUCCESSFUL SYMBOLS: {len(rows)}")

    return pd.DataFrame(rows)
