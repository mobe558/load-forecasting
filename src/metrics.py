"""Task metrics in megawatts (never accuracy): MAE, RMSE, MAPE, per-horizon MAE, bootstrap CI."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    return float(100 * np.mean(np.abs(y - p) / np.abs(y)))


def all_metrics(y, p):
    """y, p: [n_days, 24] in MW."""
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p),
            "MAE_per_h": np.abs(y - p).mean(0).round(2).tolist()}


def bootstrap_ci(per_day_err, n_boot=2000, alpha=0.05, seed=0):
    """95% CI of the mean daily error by resampling test days with replacement."""
    rng = np.random.default_rng(seed)
    n = len(per_day_err)
    means = per_day_err[rng.integers(0, n, (n_boot, n))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
