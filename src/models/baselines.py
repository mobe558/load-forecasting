"""Baselines (Core Req. 3): the reference points every deep model must beat.

* Seasonal naive (weekly): y_hat(t+h) = y(t+h-168)   "same hour, same weekday, last week"
* Seasonal naive (daily) : y_hat(t+h) = y(t+h-24)    "same hour yesterday" (shown for context)
* LightGBM (Ke et al., 2017): one gradient-boosted tree model for all 24 horizons ("pooled direct"
  strategy) - each row is one (forecast origin, horizon h) pair with hand-made lag features.
"""
import numpy as np

LGB_FEATURES = ["h", "lag_week", "lag_day", "last", "mean24", "max24", "min24", "mean168", "std24",
                "trend24"]


def seasonal_naive(src, season: int = 168):
    """Forecast = the load observed `season` hours before each target hour (all inside the inputs)."""
    t = src.origins[:, None] + np.arange(src.H)[None, :]
    return src.load[t - season]


def lgb_rows(src, cal_names):
    """Tabular rows for LightGBM: [n_origins * H, n_features]. Every feature uses only hours < t."""
    H, load, cal = src.H, src.load, src.cal
    t = src.origins
    hrs = np.arange(H)
    tgt = t[:, None] + hrs[None, :]                        # target index of each (origin, h)
    past24 = load[t[:, None] + np.arange(-24, 0)[None, :]]
    past168 = load[t[:, None] + np.arange(-168, 0)[None, :]]
    per_origin = np.stack([load[t - 1], past24.mean(1), past24.max(1), past24.min(1), past168.mean(1),
                           past24.std(1), load[t - 1] - load[t - 25]], axis=1)          # [N, 7]
    n = len(t)
    X = np.concatenate([
        np.repeat(hrs[None, :], n, 0).reshape(-1, 1),                      # h
        load[tgt - 168].reshape(-1, 1),                                    # same hour last week
        load[tgt - 24].reshape(-1, 1),                                     # same hour yesterday (h<24 -> past)
        np.repeat(per_origin, H, axis=0),
        cal[tgt].reshape(n * H, -1),                                       # calendar of the TARGET hour
    ], axis=1).astype(np.float32)
    y = load[tgt].reshape(-1)
    return X, y, LGB_FEATURES[:3] + LGB_FEATURES[3:] + list(cal_names)
