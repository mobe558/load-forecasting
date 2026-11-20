"""Inference helpers shared by the demo: load trained models and forecast the next 24 hours from
ANY 168-hour load history (MW) + its timestamps. Calendar features are rebuilt exactly as in
data/preprocess.py, scaling uses the TRAIN scaler.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from preprocess import add_calendar  # noqa: E402

from src.datasets.windows import calendar_matrix  # noqa: E402

SCALER = json.load(open(ROOT / "data" / "processed" / "scaler.json"))
L, H = 168, 24


def calendar_for(timestamps, encoding="cyclical"):
    df = add_calendar(pd.DataFrame({"timestamp": pd.to_datetime(timestamps)}))
    return calendar_matrix(df, encoding)


class DLForecaster:
    """A trained deep model (one seed directory)."""

    def __init__(self, run_dir: Path):
        import torch
        from src.models import build_model
        self.torch = torch
        cfg = json.load(open(run_dir / "config.json"))
        self.encoding = cfg.get("data", {}).get("encoding", "cyclical")
        n_cal = calendar_for(pd.date_range("2017-01-01", periods=2, freq="h"), self.encoding).shape[1]
        self.model = build_model(cfg["model"]["name"], 1 + n_cal, n_cal, H, **cfg["model"].get("params", {}))
        self.model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
        self.model.eval()

    def __call__(self, history_mw, history_ts, future_ts):
        load = (np.asarray(history_mw, dtype=np.float32) - SCALER["mean"]) / SCALER["std"]
        cp, cf = calendar_for(history_ts, self.encoding), calendar_for(future_ts, self.encoding)
        xp = self.torch.as_tensor(np.concatenate([load[:, None], cp], 1)[None], dtype=self.torch.float32)
        xf = self.torch.as_tensor(cf[None], dtype=self.torch.float32)
        with self.torch.no_grad():
            y = self.model(xp, xf).numpy()[0]
        return y * SCALER["std"] + SCALER["mean"]


class EnsembleForecaster:
    """Average of the available seeds of one experiment (what the report calls 'seed-averaged')."""

    def __init__(self, name: str):
        dirs = [d for d in sorted((ROOT / "results" / "runs" / name).glob("seed*")) if (d / "model.pt").exists()]
        if not dirs:
            raise FileNotFoundError(f"no trained checkpoints for {name} - run training first")
        self.members = [DLForecaster(d) for d in dirs]

    def __call__(self, *a):
        return np.mean([m(*a) for m in self.members], axis=0)


class LGBForecaster:
    def __init__(self, run_dir: Path):
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(run_dir / "model.txt"))

    def __call__(self, history_mw, history_ts, future_ts):
        from src.datasets.windows import ArraySource
        from src.models.baselines import lgb_rows
        load = (np.r_[np.asarray(history_mw, dtype=np.float32), np.zeros(H, np.float32)] - SCALER["mean"]) / SCALER["std"]
        cal = calendar_for(list(history_ts) + list(future_ts))
        X, _, _ = lgb_rows(ArraySource(load, cal, np.array([L]), L, H), [])
        return self.booster.predict(X) * SCALER["std"] + SCALER["mean"]


def seasonal_naive(history_mw):
    return np.asarray(history_mw[-168:-168 + 24], dtype=float)


def future_timestamps(last_ts):
    return pd.date_range(pd.Timestamp(last_ts) + pd.Timedelta("1h"), periods=H, freq="h")


def load_all():
    """Every model available on disk -> {label: callable(history_mw, history_ts, future_ts)}."""
    out = {}
    for name, label in [("transformer", "Transformer"), ("tft", "TFT"), ("lstm", "LSTM"),
                        ("transformer_finetune", "Transformer (pretrained)")]:
        try:
            out[label] = EnsembleForecaster(name)
        except (FileNotFoundError, RuntimeError):
            pass
    lgb_dir = ROOT / "results" / "runs" / "lightgbm" / "seed0"
    if (lgb_dir / "model.txt").exists():
        out["LightGBM"] = LGBForecaster(lgb_dir)
    out["Seasonal naive"] = lambda h, hts, fts: seasonal_naive(h)
    return out
