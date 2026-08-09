import numpy as np
import pandas as pd

from engine.simulator import run_simulation


def run_robustness_test(
    prices: pd.DataFrame,
    runs: int = 10,
):
    """
    Kör flera backtests från olika startpunkter
    för att kontrollera om strategin är robust.

    Varje simulering returnerar:
        equity, stats

    Robustness använder slutlig equity från varje körning.
    """

    results = []

    min_window = 60
    max_start = len(prices) - 100

    if max_start <= min_window:
        raise ValueError(f"Not enough data for robustness test: {len(prices)} rows")

    for i in range(runs):
        start = np.random.randint(
            min_window,
            max_start,
        )

        sample = prices.iloc[start:].copy()

        equity, stats = run_simulation(
            sample,
            window=60,
        )

        results.append(float(equity.iloc[-1]))

    results = np.array(results)

    return {
        "mean_return": float(results.mean() - 1),
        "std_return": float(results.std()),
        "best": float(results.max() - 1),
        "worst": float(results.min() - 1),
        "runs": results.tolist(),
    }
