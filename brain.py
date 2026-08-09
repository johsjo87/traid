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
from engine.prediction_engine import predict_from_history
from engine.benchmark import benchmark_stats
from engine.validation import validate_no_lookahead


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
        "SPY",
    ]

    # -------------------------
    # LOAD DATA
    # -------------------------

    prices = load_price_history(tickers)

    print(f"DATA: {len(prices)} rows, {len(prices.columns)} symbols")

    # -------------------------
    # RESEARCH DATASET
    # -------------------------

    dataset = build_feature_dataset(prices)

    print(f"\nRESEARCH DATASET: {len(dataset)} rows")

    # -------------------------
    # LOOKAHEAD VALIDATION
    # -------------------------

    validation = validate_no_lookahead(prices)

    print("\nLOOKAHEAD VALIDATION")
    print("--------------------------------")

    print(f"Valid: {validation['valid']}")

    if "rows_checked" in validation:
        print(f"Rows checked: {validation['rows_checked']}")

    if "violation_count" in validation:
        print(f"Violations: {validation['violation_count']}")

    if validation.get("violations"):
        print("\nViolations:")

        for violation in validation["violations"]:
            print("-", violation)

    if not validation["valid"]:
        raise ValueError("Lookahead validation failed. Backtest should not be trusted.")

    # -------------------------
    # STRATEGY
    # -------------------------

    result = build_strategy(
        prices,
        dataset,
        top_n=10,
        use_prediction=True,
    )

    # -------------------------
    # MARKET OUTLOOK
    # -------------------------

    outlook = build_market_outlook(
        result["features"],
        result["regime"],
    )

    risk = analyze_risk(
        result["features"],
        result["regime"],
        result["portfolio"],
    )

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

    # -------------------------
    # MARKET RISK
    # -------------------------

    print("\nMARKET RISK")
    print("--------------------------------")

    print(f"Risk:              {risk['risk']}")

    print(f"Suggested exposure:{risk['exposure']}")

    print("\nReasons:")

    for reason in risk["reasons"]:
        print("-", reason)

    # -------------------------
    # PORTFOLIO
    # -------------------------

    print("\nPORTFOLIO")
    print("--------------------------------")

    print(
        result["portfolio"][
            [
                "symbol",
                "alpha_score",
                "prediction_bonus",
                "confidence",
                "prediction_samples",
                "score",
                "signal",
                "weight",
            ]
        ].to_string(index=False)
    )

    # -------------------------
    # STOCK ANALYSIS
    # -------------------------

    stock_reports = analyze_portfolio(
        result["portfolio"],
        result["features"],
    )

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
        result["portfolio"],
        result["features"],
    )

    print("\nTIMEFRAME ANALYSIS")
    print("--------------------------------")

    for item in timeframe_reports:
        print("\n" + item["symbol"])

        print("Short term:", item["short_term"])

        print("Medium term:", item["medium_term"])

        print("Long term:", item["long_term"])

    # -------------------------
    # PREDICTION ENGINE
    # -------------------------

    predictions = predict_from_history(
        result["features"],
        dataset,
    )

    print("\nPREDICTION ENGINE")
    print("--------------------------------")

    for _, row in predictions.iterrows():
        print(f"\n{row['symbol']}")

        if row["samples"] == 0:
            print("No similar historical cases found.")

            continue

        print(f"Setup quality: {row['setup_quality']}/100")

        print(f"Prediction score: {row['prediction_score']:+.2f}")

        print(f"Reliability: {row['reliability']}")

        print(f"Average future return: {row['avg_future_return']:.2%}")

        print(f"Positive outcomes: {row['positive_rate']:.1f}%")

        print(f"Historical samples: {int(row['samples'])}")

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
    # BACKTEST A/B TEST
    # -------------------------

    print("\nBACKTEST A/B TEST")
    print("--------------------------------")

    # --------------------------------------------------
    # A: ALPHA + PREDICTION
    # --------------------------------------------------

    equity_prediction, stats_prediction = run_simulation(
        prices,
        window=60,
        use_prediction=True,
    )

    # --------------------------------------------------
    # B: ALPHA ONLY
    # --------------------------------------------------

    equity_alpha, stats_alpha = run_simulation(
        prices,
        window=60,
        use_prediction=False,
    )

    # --------------------------------------------------
    # SPY BENCHMARK
    # --------------------------------------------------

    spy = benchmark_stats(prices["SPY"].dropna())

    # --------------------------------------------------
    # ALPHA + PREDICTION
    # --------------------------------------------------

    print("\nTRAID — ALPHA + PREDICTION")
    print("--------------------------------")

    print(f"Return:           {stats_prediction['total_return']:.2%}")

    print(f"Max drawdown:     {stats_prediction['max_drawdown']:.2%}")

    print(f"Trades:           {stats_prediction['trades']}")

    print(f"Win rate:         {stats_prediction['win_rate']:.2%}")

    print(f"Profit factor:    {stats_prediction['profit_factor']:.2f}")

    # --------------------------------------------------
    # ALPHA ONLY
    # --------------------------------------------------

    print("\nTRAID — ALPHA ONLY")
    print("--------------------------------")

    print(f"Return:           {stats_alpha['total_return']:.2%}")

    print(f"Max drawdown:     {stats_alpha['max_drawdown']:.2%}")

    print(f"Trades:           {stats_alpha['trades']}")

    print(f"Win rate:         {stats_alpha['win_rate']:.2%}")

    print(f"Profit factor:    {stats_alpha['profit_factor']:.2f}")

    # --------------------------------------------------
    # SPY
    # --------------------------------------------------

    print("\nSPY")
    print("--------------------------------")

    print(f"Return:           {spy['total_return']:.2%}")

    print(f"Max drawdown:     {spy['max_drawdown']:.2%}")

    # --------------------------------------------------
    # OUTPERFORMANCE
    # --------------------------------------------------

    print("\nOUTPERFORMANCE")
    print("--------------------------------")

    print(
        f"Alpha + Prediction: "
        f"{stats_prediction['total_return'] - spy['total_return']:+.2%}"
    )

    print(
        f"Alpha Only:         {stats_alpha['total_return'] - spy['total_return']:+.2%}"
    )

    # --------------------------------------------------
    # PREDICTION CONTRIBUTION
    # --------------------------------------------------

    print("\nPREDICTION CONTRIBUTION")
    print("--------------------------------")

    print(
        f"Return difference:  "
        f"{stats_prediction['total_return'] - stats_alpha['total_return']:+.2%}"
    )

    print(
        f"Drawdown difference:"
        f" {stats_prediction['max_drawdown'] - stats_alpha['max_drawdown']:+.2%}"
    )

    print(
        f"Win rate difference:"
        f" {stats_prediction['win_rate'] - stats_alpha['win_rate']:+.2%}"
    )

    print(
        f"Profit factor diff: "
        f"{stats_prediction['profit_factor'] - stats_alpha['profit_factor']:+.2f}"
    )

    # -------------------------
    # ROBUSTNESS
    # -------------------------

    try:
        robustness = run_robustness_test(
            prices,
            runs=10,
        )

        print("\nROBUSTNESS")
        print("--------------------------------")

        print(robustness)

    except Exception as e:
        print("\nROBUSTNESS SKIPPED:")
        print(e)


if __name__ == "__main__":
    main()
