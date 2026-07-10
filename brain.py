from engine.historical import load_price_history
from engine.strategy import build_strategy
from engine.simulator import run_simulation
from engine.robustness import run_robustness_test
from engine.feature_analysis import analyze_features
from engine.feature_dataset import build_feature_dataset
from engine.feature_optimizer import optimize_features
from engine.market_outlook import build_market_outlook
from engine.risk_engine import analyze_risk
from engine.stock_analysis import analyze_portfolio
from engine.timeframe_analysis import analyze_timeframes_portfolio


def main():

    tickers = [
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
        "JPM",
        "BAC",
        "GS",
        "MS",
        "LLY",
        "JNJ",
        "UNH",
        "ABBV",
        "KO",
        "MCD",
    ]

    prices = load_price_history(tickers)

    print(f"DATA: {len(prices)} rows, {len(prices.columns)} symbols")

    dataset = build_feature_dataset(prices)

    print(f"\nRESEARCH DATASET: {len(dataset)} rows")

    result = build_strategy(prices, top_n=10)

    outlook = build_market_outlook(result["features"], result["regime"])

    risk = analyze_risk(result["features"], result["regime"])

    print("\nREGIME:", result["regime"])

    print("\nMARKET OUTLOOK")
    print("--------------------------------")

    print(f"Regime:            {outlook['regime']}")

    print(f"Short-term:        {outlook['short_term']}")

    print(f"Risk level:        {outlook['risk']}")

    print(f"Momentum:          {outlook['momentum']:.4f}")

    print(f"Relative strength: {outlook['relative_strength']:.4f}")

    print(f"Volatility:        {outlook['volatility']:.4f}")

    print(f"Distance to high:  {outlook['distance_high']:.4f}")

    print("\nAnalysis:")
    print(outlook["analysis"])

    print("\nMARKET RISK")
    print("--------------------------------")

    print(f"Risk:              {risk['risk']}")

    print(f"Suggested exposure:{risk['exposure']}")

    print("\nReasons:")

    for reason in risk["reasons"]:
        print("-", reason)

    print("\nPORTFOLIO")
    print("--------------------------------")

    print(result["portfolio"].to_string(index=False))

    # -------------------------
    # STOCK ANALYSIS
    # -------------------------

    stock_reports = analyze_portfolio(result["portfolio"], result["features"])

    print("\nSTOCK ANALYSIS")
    print("--------------------------------")

    for report in stock_reports:
        print("\n" + report["symbol"])

        print("Score:", report["score"])

        print("Signal:", report["signal"])

        print("Short term:", report["outlook"])

        print("Strengths:")

        for item in report["strengths"]:
            print("-", item)

        print("Risks:")

        for item in report["risks"]:
            print("-", item)

    # -------------------------
    # TIMEFRAME ANALYSIS
    # -------------------------

    timeframe_reports = analyze_timeframes_portfolio(
        result["portfolio"], result["features"]
    )

    print("\nTIMEFRAME ANALYSIS")
    print("--------------------------------")

    for item in timeframe_reports:
        print("\n" + item["symbol"])

        print("Short term:", item["short_term"])

        print("Medium term:", item["medium_term"])

        print("Long term:", item["long_term"])

    # -------------------------
    # FEATURE ANALYSIS
    # -------------------------

    feature_summary = analyze_features(dataset)

    print("\nFEATURE PREDICTIVE POWER")
    print("--------------------------------")

    print(feature_summary.to_string(index=False))

    # -------------------------
    # FEATURE OPTIMIZATION
    # -------------------------

    optimization = optimize_features(dataset)

    print("\nFEATURE OPTIMIZATION")
    print("--------------------------------")

    print(optimization.to_string(index=False))

    # -------------------------
    # SIMULATION
    # -------------------------

    equity = run_simulation(prices, window=60)

    print("\nFINAL EQUITY:", equity.iloc[-1])

    # -------------------------
    # ROBUSTNESS
    # -------------------------

    try:
        stats = run_robustness_test(prices, runs=10)

        print("\nROBUSTNESS")
        print("--------------------------------")

        print(stats)

    except Exception as e:
        print("\nROBUSTNESS SKIPPED:")
        print(e)


if __name__ == "__main__":
    main()
