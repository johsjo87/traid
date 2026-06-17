from engine.historical import load_price_history
from engine.strategy import build_strategy
from engine.simulator import run_simulation
from engine.robustness import run_robustness_test
from engine.feature_analysis import analyze_features


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

    # -------------------------
    # LOAD DATA
    # -------------------------

    prices = load_price_history(tickers)

    print(f"DATA: {len(prices)} rows, {len(prices.columns)} symbols")

    # -------------------------
    # BUILD STRATEGY
    # -------------------------

    result = build_strategy(prices, top_n=10)

    # -------------------------
    # FEATURE ANALYSIS
    # -------------------------

    feature_summary = analyze_features(result["features"])

    # -------------------------
    # OUTPUT
    # -------------------------

    print("\nREGIME:", result["regime"])

    print("\nPORTFOLIO")
    print("--------------------------------")
    print(result["portfolio"].to_string(index=False))

    print("\nFEATURE ANALYSIS")
    print("--------------------------------")
    print(feature_summary.to_string(index=False))

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
