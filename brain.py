from engine.historical import load_price_history
from engine.strategy import build_strategy
from engine.simulator import run_simulation
from engine.robustness import run_robustness_test


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

    result = build_strategy(prices, top_n=10)

    print("\nREGIME:", result["regime"])
    print(result["portfolio"].to_string(index=False))

    equity = run_simulation(prices, window=60)

    print("\nFINAL EQUITY:", equity.iloc[-1])

    try:
        stats = run_robustness_test(prices, runs=10)

        print("\nROBUSTNESS:")
        print(stats)

    except Exception as e:
        print("ROBUSTNESS SKIPPED:", e)


if __name__ == "__main__":
    main()
