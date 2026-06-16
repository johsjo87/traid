import numpy as np
import pandas as pd

from engine.simulator import run_simulation


def run_robustness_test(prices: pd.DataFrame, runs: int = 10):

    results = []

    min_window = 60

    max_start = len(prices) - 100

    if max_start <= min_window:
        raise ValueError(f"Not enough data for robustness test: {len(prices)} rows")

    for i in range(runs):
        start = np.random.randint(min_window, max_start)
        sample = prices.iloc[start:]

        equity = run_simulation(sample, window=60)

        results.append(equity.iloc[-1])

    results = np.array(results)

    return {
        "mean_return": float(results.mean()),
        "std_return": float(results.std()),
        "best": float(results.max()),
        "worst": float(results.min()),
        "runs": results.tolist(),
    }
