from engine.features import build_feature_matrix
from engine.regime import detect_regime
from engine.alpha import build_alpha
from engine.portfolio import build_portfolio_weights
from engine.backtest import backtest_signals
from engine.simulator import run_simulation


def main():

    print("\n" + "=" * 70)
    print("TRAID V8 - HYBRID ENGINE")
    print("=" * 70)

    # -------------------------
    # REAL-TIME PIPELINE (V7)
    # -------------------------
    features = build_feature_matrix()
    regime = detect_regime(features)

    alpha = build_alpha(features, regime)
    portfolio = build_portfolio_weights(alpha, top_n=10)

    print("\nREGIME:")
    print(regime)

    print("\nPORTFOLIO:")
    print(portfolio.to_string(index=False))

    bt = backtest_signals(portfolio)

    print("\nBACKTEST METRICS (V7)")
    print("------------------------------------")
    print(bt.to_string(index=False))

    # -------------------------
    # SIMULATION PIPELINE (V8)
    # -------------------------
    equity = run_simulation(30)

    print("\nV8 SIMULATION")
    print("------------------------------------")
    print(f"Final equity: {equity.iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
