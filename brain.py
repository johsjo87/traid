from engine.historical import load_price_history
from engine.features import build_feature_matrix
from engine.regime import detect_regime
from engine.alpha import build_alpha
from engine.portfolio import build_portfolio_weights
from engine.backtest import backtest_signals
from engine.simulator import run_simulation


def main():

    print("\n" + "=" * 70)
    print("TRAID V9 - CLEAN HYBRID ENGINE")
    print("=" * 70)

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

    features = build_feature_matrix(prices)
    regime = detect_regime(features)

    alpha = build_alpha(features, regime)
    portfolio = build_portfolio_weights(alpha, top_n=10)

    print("\nREGIME:")
    print(regime)

    print("\nPORTFOLIO:")
    print(portfolio.to_string(index=False))

    bt = backtest_signals(portfolio)

    print("\nBACKTEST METRICS")
    print(bt.to_string(index=False))

    equity = run_simulation(30)
    print("\nFINAL EQUITY:", equity.iloc[-1])


if __name__ == "__main__":
    main()
