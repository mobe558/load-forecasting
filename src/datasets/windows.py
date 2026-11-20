"""Sliding-window samples for multi-horizon forecasting (shared by every model).

One sample, with forecast origin t (= first hour to predict):
    x_past   : [lookback, 1 + C]  scaled load + calendar features for hours t-168 ... t-1
    x_future : [horizon,  C]      calendar features for hours t ... t+23 (known in advance)
    y        : [horizon]          scaled load for hours t ... t+23   (the target)

Leakage rule: a sample belongs to split S only if ALL of its 24 target hours are in S.
Its input hours may lie in an earlier split - that is genuine past information which a
forecaster would also have in real life - but never in a later one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))

CYCLICAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "doy_sin", "doy_cos", "is_weekend", "is_holiday"]


def load_processed() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "pjme_processed.csv", parse_dates=["timestamp"])


def calendar_matrix(df: pd.DataFrame, encoding: str = "cyclical") -> np.ndarray:
    """Calendar features as a matrix. 'onehot' is the alternative representation
    for the data-strategy study (Core Req. 7)."""
    if encoding == "cyclical":
        return df[CYCLICAL].to_numpy(np.float32)
    if encoding == "onehot":
        parts = [np.eye(24)[df["hour"]], np.eye(7)[df["dow"]], np.eye(12)[df["month"] - 1],
                 df[["is_weekend", "is_holiday"]].to_numpy()]
        return np.concatenate(parts, axis=1).astype(np.float32)
    if encoding == "none":                       # ablation: no calendar information at all
        return np.zeros((len(df), 0), np.float32)
    raise ValueError(encoding)


class LoadWindows:
    def __init__(self, df: pd.DataFrame, split: str, encoding: str | None = None,
                 lookback: int | None = None, horizon: int | None = None, stride: int | None = None):
        w = CFG["windowing"]
        self.L = lookback or w["lookback"]
        self.H = horizon or w["horizon"]
        self.split = split
        stride = stride or (w["train_stride"] if split == "train" else w["eval_stride"])
        enc = encoding or CFG["features"]["calendar_encoding"]

        self.timestamps = df["timestamp"].to_numpy()
        self.load = df["load_scaled"].to_numpy(np.float32)
        self.cal = calendar_matrix(df, enc)
        in_split = (df["split"] == split).to_numpy()

        # candidate origins t: enough history behind, all H target hours inside this split
        n = len(df)
        t = np.arange(self.L, n - self.H + 1)
        ok = np.array([in_split[i] and in_split[i + self.H - 1] for i in t])  # split is contiguous
        t = t[ok]
        if split != "train":                    # day-ahead protocol: one forecast per day at 00:00
            t = t[df["hour"].to_numpy()[t] == 0]
        self.origins = t[::stride] if split == "train" else t[:: max(1, stride // 24)]

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int):
        t = self.origins[i]
        past = slice(t - self.L, t)
        fut = slice(t, t + self.H)
        x_past = np.concatenate([self.load[past, None], self.cal[past]], axis=1)
        return x_past, self.cal[fut], self.load[fut]

    def arrays(self):
        """All samples stacked - handy for the classical-ML baseline (small splits only)."""
        xs = [self[i] for i in range(len(self))]
        return tuple(np.stack(a) for a in zip(*xs))

    def origin_times(self) -> np.ndarray:
        return self.timestamps[self.origins]


def check_no_leakage(df: pd.DataFrame) -> dict:
    """Assertions proving the split is leakage-free. Run by experiments/check_split.py."""
    out = {}
    order = {"train": 0, "val": 1, "test": 2}
    ts = df["timestamp"].to_numpy()
    for split in ["train", "val", "test"]:
        w = LoadWindows(df, split)
        tgt_first, tgt_last = w.origins, w.origins + w.H - 1
        inp_first = w.origins - w.L
        # 1) every target hour is inside the split
        assert (df["split"].to_numpy()[tgt_first] == split).all()
        assert (df["split"].to_numpy()[tgt_last] == split).all()
        # 2) no input hour comes from a LATER split
        later = [s for s, k in order.items() if k > order[split]]
        assert not np.isin(df["split"].to_numpy()[w.origins - 1], later).any()
        out[split] = {"samples": len(w), "first_origin": str(ts[tgt_first[0]]),
                      "last_target": str(ts[tgt_last[-1]]),
                      "earliest_input": str(ts[inp_first[0]])}
    # 3) target hours of different splits never overlap
    tr, va, te = (LoadWindows(df, s) for s in ["train", "val", "test"])
    assert ts[tr.origins[-1] + tr.H - 1] < ts[va.origins[0]]
    assert ts[va.origins[-1] + va.H - 1] < ts[te.origins[0]]
    # 4) scaler was fitted on train only
    import json
    sc = json.load(open(ROOT / "data" / "processed" / "scaler.json"))
    train_load = df.loc[df["split"] == "train", "load_mw"]
    assert abs(sc["mean"] - train_load.mean()) < 1e-6 and sc["fitted_on"] == "train"
    out["scaler_fitted_on"] = sc["fitted_on"]
    return out


# ----------------------------------------------------------------------------------------
# Fast batching for PyTorch: windows are cut on the fly with index arithmetic, so the
# 113k overlapping training windows never have to be stored in memory (~0.8 GB otherwise).
# ----------------------------------------------------------------------------------------
class ArraySource:
    """Minimal container (load, cal, origins, L, H) - used to merge several series."""

    def __init__(self, load, cal, origins, L, H):
        self.load, self.cal, self.origins, self.L, self.H = load, cal, np.asarray(origins), L, H

    def __len__(self):
        return len(self.origins)


def concat_sources(sources):
    """Concatenate several series end to end; origins are shifted so no window crosses a border."""
    loads, cals, origins, off = [], [], [], 0
    for s in sources:
        loads.append(s.load), cals.append(s.cal), origins.append(s.origins + off)
        off += len(s.load)
    return ArraySource(np.concatenate(loads), np.concatenate(cals), np.concatenate(origins),
                       sources[0].L, sources[0].H)


def restrict_origins(src, df, start=None):
    """Keep only training windows whose forecast origin is >= start (limited-data experiment)."""
    if start is not None:
        keep = df["timestamp"].to_numpy()[src.origins] >= np.datetime64(pd.Timestamp(start))
        src.origins = src.origins[keep]
    return src


class TorchBatcher:
    def __init__(self, src):
        import torch
        self.torch = torch
        self.load = torch.as_tensor(src.load, dtype=torch.float32)
        self.cal = torch.as_tensor(src.cal, dtype=torch.float32)
        self.origins = torch.as_tensor(src.origins, dtype=torch.long)
        self.past = torch.arange(-src.L, 0)
        self.fut = torch.arange(src.H)

    def __len__(self):
        return len(self.origins)

    def make(self, t):
        pi, fi = t[:, None] + self.past, t[:, None] + self.fut
        x_past = self.torch.cat([self.load[pi][..., None], self.cal[pi]], dim=-1)
        return x_past, self.cal[fi], self.load[fi]

    def batches(self, batch_size, shuffle=False, n_samples=None, generator=None):
        torch = self.torch
        if shuffle:
            order = torch.randperm(len(self.origins), generator=generator)
            if n_samples:
                order = order[:n_samples]
        else:
            order = torch.arange(len(self.origins))
        for i in range(0, len(order), batch_size):
            yield self.make(self.origins[order[i:i + batch_size]])
