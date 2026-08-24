import pandas as pd
import numpy as np

from engine.features import build_feature_matrix
from engine.alpha import build_alpha
from engine.prediction_engine import predict_from_history
from engine.scoring import combine_scores
from engine.portfolio import build_portfolio_weights


def _bucket_alpha(score):
    if score < 20:
        return "0-20"
    if score < 40:
        return "20-40"
    if score < 60:
        return "40-60"
    if score < 80:
        return "60-80"
    return "80-100"


def _bucket_prediction(score):
    if score < -5:
        return "<-5"
    if score < 0:
        return "-5 to 0"
    if score < 2:
        return "0 to 2"
    if score < 5:
        return "2 to 5"
    return "5+"


def _bucket_quality(score):
    if score < 40:
        return "0-40"
    if score < 60:
        return "40-60"
    if score < 80:
        return "60-80"
    return "80-100"


def _summarize(df, group_column):
    if df.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "observations",
                "avg_return",
                "median_return",
                "win_rate",
            ]
        )

    result = (
        df.groupby(group_column, dropna=False)
        .agg(
            observations=("future_return", "size"),
            avg_return=("future_return", "mean"),
            median_return=("future_return", "median"),
            win_rate=("future_return", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    return result


def _calculate_future_return(
    prices,
    symbol,
    decision_date,
    horizon,
):
    if symbol not in prices.columns:
        return None

    symbol_prices = prices[symbol].dropna()

    if symbol_prices.empty:
        return None

    historical_prices = symbol_prices.loc[symbol_prices.index <= decision_date]

    if historical_prices.empty:
        return None

    decision_position = len(historical_prices) - 1
    future_position = decision_position + horizon

    if future_position >= len(symbol_prices):
        return None

    p0 = float(historical_prices.iloc[-1])
    p1 = float(symbol_prices.iloc[future_position])

    if p0 <= 0:
        return None

    return (p1 / p0) - 1


def _build_historical_datasets(
    prices,
    lookback,
    horizon,
):
    """
    Builds one research dataset for every historical
    decision date.

    Every dataset contains only information available
    at that decision date plus the future return target.
    """

    historical = {}

    for day in range(lookback, len(prices)):
        history_prices = prices.iloc[:day]

        if history_prices.empty:
            continue

        decision_date = history_prices.index[-1]

        features = build_feature_matrix(history_prices)

        if features is None or features.empty:
            continue

        rows = []

        for _, feature in features.iterrows():
            symbol = feature["symbol"]

            future_return = _calculate_future_return(
                prices,
                symbol,
                decision_date,
                horizon,
            )

            if future_return is None:
                continue

            row = feature.to_dict()

            row["decision_date"] = decision_date
            row["future_return"] = future_return

            rows.append(row)

        if rows:
            historical[decision_date] = pd.DataFrame(rows)

    return historical


def _calculate_portfolio_result(
    portfolio,
    current_dataset,
    horizon,
):
    """
    Calculates the realized return of a portfolio created
    at one historical decision point.

    The portfolio weights are determined only from information
    available at the decision date.
    """

    if portfolio is None or portfolio.empty:
        return None

    returns = []

    for _, position in portfolio.iterrows():
        symbol = position["symbol"]

        matches = current_dataset[current_dataset["symbol"] == symbol]

        if matches.empty:
            continue

        future_return = float(matches.iloc[0]["future_return"])

        weight = float(position.get("weight", 0.0))

        returns.append(
            {
                "symbol": symbol,
                "weight": weight,
                "future_return": future_return,
                "weighted_return": weight * future_return,
            }
        )

    if not returns:
        return None

    result = pd.DataFrame(returns)

    total_weight = result["weight"].sum()

    if total_weight <= 0:
        return None

    # Normalize in case filtering removed a position.
    result["normalized_weight"] = result["weight"] / total_weight

    portfolio_return = (result["normalized_weight"] * result["future_return"]).sum()

    return {
        "portfolio_return": float(portfolio_return),
        "positions": int(len(result)),
    }


def _calculate_metrics(returns):
    if returns is None or len(returns) == 0:
        return {
            "return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "observations": 0,
        }

    returns = pd.Series(returns).astype(float)

    equity = (1 + returns).cumprod()

    total_return = float(equity.iloc[-1] - 1)

    peak = equity.cummax()

    drawdown = equity / peak - 1

    max_drawdown = float(drawdown.min())

    volatility = returns.std()

    if volatility > 0:
        sharpe = float(returns.mean() / volatility * np.sqrt(252 / 20))
    else:
        sharpe = 0.0

    win_rate = float((returns > 0).mean())

    return {
        "return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "observations": len(returns),
    }


def _build_portfolio_validation(
    decision_results,
):
    """
    Converts historical decision results into portfolio-level
    walk-forward metrics.
    """

    if not decision_results:
        return {
            "alpha_prediction": {},
            "alpha_only": {},
        }

    results = pd.DataFrame(decision_results)

    alpha_prediction = _calculate_metrics(results["portfolio_return"])

    alpha_only = _calculate_metrics(results["alpha_only_return"])

    return {
        "alpha_prediction": alpha_prediction,
        "alpha_only": alpha_only,
    }


def validate_decisions(
    prices: pd.DataFrame,
    lookback: int = 60,
    horizon: int = 20,
    max_samples: int = 100,
):

    empty = pd.DataFrame()

    if prices is None or prices.empty:
        return {
            "decisions": empty,
            "alpha": empty,
            "prediction": empty,
            "reliability": empty,
            "setup_quality": empty,
            "portfolio_validation": {},
            "portfolio_history": empty,
        }

    if lookback <= 0:
        raise ValueError("lookback must be greater than 0")

    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    # ---------------------------------------------------------
    # BUILD HISTORICAL DATA
    # ---------------------------------------------------------

    historical = _build_historical_datasets(
        prices,
        lookback,
        horizon,
    )

    if not historical:
        return {
            "decisions": empty,
            "alpha": empty,
            "prediction": empty,
            "reliability": empty,
            "setup_quality": empty,
            "portfolio_validation": {},
            "portfolio_history": empty,
        }

    decision_dates = sorted(historical.keys())

    decisions = []
    portfolio_results = []

    # ---------------------------------------------------------
    # WALK-FORWARD REPLAY
    # ---------------------------------------------------------

    for date_index, decision_date in enumerate(decision_dates):
        current_dataset = historical[decision_date]

        if current_dataset.empty:
            continue

        # -----------------------------------------------------
        # CURRENT FEATURES
        # -----------------------------------------------------

        current_features = current_dataset.drop(
            columns=[
                "future_return",
                "decision_date",
            ],
            errors="ignore",
        ).copy()

        if current_features.empty:
            continue

        # -----------------------------------------------------
        # ALPHA
        # -----------------------------------------------------

        alpha = build_alpha(
            current_features,
            regime="NEUTRAL",
        )

        if alpha is None or alpha.empty:
            continue

        # -----------------------------------------------------
        # HISTORICAL DATA ONLY
        # -----------------------------------------------------

        previous_datasets = []

        for previous_date in decision_dates[:date_index]:
            previous_dataset = historical[previous_date]

            if previous_dataset.empty:
                continue

            previous_datasets.append(previous_dataset)

        if not previous_datasets:
            continue

        history = pd.concat(
            previous_datasets,
            ignore_index=True,
        )

        if history.empty:
            continue

        # -----------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------

        predictions = predict_from_history(
            current_features,
            history,
            max_samples=max_samples,
        )

        if predictions is None or predictions.empty:
            continue

        # -----------------------------------------------------
        # ALPHA + PREDICTION
        # -----------------------------------------------------

        combined = combine_scores(
            alpha,
            predictions,
        )

        if combined.empty:
            continue

        # -----------------------------------------------------
        # ALPHA + PREDICTION PORTFOLIO
        # -----------------------------------------------------

        prediction_portfolio = build_portfolio_weights(
            combined,
            regime="NEUTRAL",
        )

        prediction_result = _calculate_portfolio_result(
            prediction_portfolio,
            current_dataset,
            horizon,
        )

        # -----------------------------------------------------
        # ALPHA ONLY PORTFOLIO
        # -----------------------------------------------------

        alpha_only = alpha.copy()

        alpha_only["prediction_bonus"] = 0.0
        alpha_only["prediction_samples"] = 0
        alpha_only["confidence"] = 0.0
        alpha_only["alpha_score"] = alpha_only["score"]
        alpha_only["final_score"] = alpha_only["score"]

        alpha_portfolio = build_portfolio_weights(
            alpha_only,
            regime="NEUTRAL",
        )

        alpha_result = _calculate_portfolio_result(
            alpha_portfolio,
            current_dataset,
            horizon,
        )

        if prediction_result is None or alpha_result is None:
            continue

        # -----------------------------------------------------
        # STORE PORTFOLIO RESULT
        # -----------------------------------------------------

        portfolio_results.append(
            {
                "decision_date": decision_date,
                "portfolio_return": prediction_result["portfolio_return"],
                "alpha_only_return": alpha_result["portfolio_return"],
                "prediction_positions": prediction_result["positions"],
                "alpha_only_positions": alpha_result["positions"],
            }
        )

        # -----------------------------------------------------
        # STORE INDIVIDUAL DECISIONS
        # -----------------------------------------------------

        for _, row in combined.iterrows():
            symbol = row["symbol"]

            future_return = _calculate_future_return(
                prices,
                symbol,
                decision_date,
                horizon,
            )

            if future_return is None:
                continue

            prediction_row = predictions[predictions["symbol"] == symbol]

            if prediction_row.empty:
                continue

            prediction_row = prediction_row.iloc[0]

            decisions.append(
                {
                    "decision_date": decision_date,
                    "symbol": symbol,
                    "alpha_score": float(
                        row.get(
                            "alpha_score",
                            row["score"],
                        )
                    ),
                    "prediction_score": float(
                        row.get(
                            "prediction_score",
                            0.0,
                        )
                    ),
                    "prediction_bonus": float(
                        row.get(
                            "prediction_bonus",
                            0.0,
                        )
                    ),
                    "confidence": float(
                        row.get(
                            "confidence",
                            0.0,
                        )
                    ),
                    "prediction_samples": int(
                        row.get(
                            "prediction_samples",
                            0,
                        )
                    ),
                    "setup_quality": float(
                        prediction_row.get(
                            "setup_quality",
                            0.0,
                        )
                    ),
                    "reliability": prediction_row.get(
                        "reliability",
                        "LOW",
                    ),
                    "final_score": float(row["final_score"]),
                    "future_return": float(future_return),
                }
            )

    # ---------------------------------------------------------
    # NO DECISIONS
    # ---------------------------------------------------------

    if not decisions:
        return {
            "decisions": empty,
            "alpha": empty,
            "prediction": empty,
            "reliability": empty,
            "setup_quality": empty,
            "portfolio_validation": _build_portfolio_validation(portfolio_results),
            "portfolio_history": pd.DataFrame(portfolio_results),
        }

    df = pd.DataFrame(decisions)

    # ---------------------------------------------------------
    # BUCKETS
    # ---------------------------------------------------------

    df["alpha_bucket"] = df["alpha_score"].apply(_bucket_alpha)

    df["prediction_bucket"] = df["prediction_score"].apply(_bucket_prediction)

    df["quality_bucket"] = df["setup_quality"].apply(_bucket_quality)

    # ---------------------------------------------------------
    # SUMMARIES
    # ---------------------------------------------------------

    alpha_summary = _summarize(
        df,
        "alpha_bucket",
    )

    prediction_summary = _summarize(
        df,
        "prediction_bucket",
    )

    reliability_summary = _summarize(
        df,
        "reliability",
    )

    quality_summary = _summarize(
        df,
        "quality_bucket",
    )

    # ---------------------------------------------------------
    # PORTFOLIO VALIDATION
    # ---------------------------------------------------------

    portfolio_validation = _build_portfolio_validation(portfolio_results)

    portfolio_history = pd.DataFrame(portfolio_results)

    return {
        "decisions": df,
        "alpha": alpha_summary,
        "prediction": prediction_summary,
        "reliability": reliability_summary,
        "setup_quality": quality_summary,
        "portfolio_validation": portfolio_validation,
        "portfolio_history": portfolio_history,
    }
