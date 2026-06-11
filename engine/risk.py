import pandas as pd


TECH = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "INTC",
    "QCOM",
    "AVGO",
    "ORCL",
    "CRM",
    "ADBE",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "NFLX",
}

FINANCIALS = {
    "JPM",
    "BAC",
    "GS",
    "MS",
    "V",
    "MA",
}

HEALTHCARE = {
    "LLY",
    "JNJ",
    "UNH",
    "ABBV",
    "PFE",
}

INDUSTRIALS = {
    "CAT",
    "GE",
    "HON",
    "DE",
}

CONSUMER = {
    "COST",
    "WMT",
    "KO",
    "PEP",
    "MCD",
}

ENERGY = {
    "XOM",
    "CVX",
    "COP",
}

ETF = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
}


def apply_risk_adjustment(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    penalties = []

    for symbol in df["symbol"]:
        penalty = 1.00

        if symbol in TECH:
            penalty *= 0.95

        if symbol in ETF:
            penalty *= 0.98

        penalties.append(penalty)

    df["risk_multiplier"] = penalties

    df["score"] = df["score"] * df["risk_multiplier"]

    return df
