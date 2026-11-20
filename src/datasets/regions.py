"""Other PJM regions used ONLY for pretraining (pretrained-vs-scratch study, Core Req. 7).

Leakage guard: only hours up to the PJME train_end (2014-12-31 23:00) are used, so the pretraining
corpus never overlaps the validation/test period in time (regions share weather, so later data from
a neighbouring region would leak information about the test period). Each region is z-scored with
its own statistics from that same period.
"""
import sys

import numpy as np
import pandas as pd

from .windows import CFG, ROOT, ArraySource, calendar_matrix, concat_sources

sys.path.insert(0, str(ROOT / "data"))
from preprocess import add_calendar  # noqa: E402

REGIONS = ["AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "NI", "PJMW"]
BASE_URL = "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/{}_hourly.csv"


def region_path(r):
    return ROOT / "data" / "raw" / "regions" / f"{r}_hourly.csv"


def load_region(r: str) -> pd.DataFrame:
    df = pd.read_csv(region_path(r), parse_dates=["Datetime"])
    df = df.rename(columns={"Datetime": "timestamp", f"{r}_MW": "load_mw"}).sort_values("timestamp")
    df = df.groupby("timestamp", as_index=False)["load_mw"].mean()          # DST duplicates
    grid = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(grid).rename_axis("timestamp")
    df["load_mw"] = df["load_mw"].interpolate(limit=3)
    df = df.reset_index()
    df = df[df["timestamp"] <= pd.Timestamp(CFG["split"]["train_end"])]      # leakage guard
    df = df.dropna().reset_index(drop=True)
    return add_calendar(df)


def pretrain_source(encoding: str, stride: int = 1):
    L, H = CFG["windowing"]["lookback"], CFG["windowing"]["horizon"]
    sources = []
    for r in REGIONS:
        df = load_region(r)
        if len(df) < L + H + 24:
            continue
        x = df["load_mw"].to_numpy()
        load = ((x - x.mean()) / x.std()).astype(np.float32)
        ts = df["timestamp"]
        # only windows whose hours are all consecutive (a few regions have longer gaps -> dropna)
        consecutive = (ts.diff().dt.total_seconds().fillna(3600) == 3600).to_numpy()
        breaks = np.cumsum(~consecutive)
        t = np.arange(L, len(df) - H + 1, stride)
        ok = breaks[t - L] == breaks[t + H - 1]
        sources.append(ArraySource(load, calendar_matrix(df, encoding), t[ok], L, H))
    return concat_sources(sources)
