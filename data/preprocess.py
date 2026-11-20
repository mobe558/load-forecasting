"""Clean the raw PJME file, add calendar features, assign the chronological
split and fit the scaler on TRAIN ONLY.

Usage:  python data/preprocess.py
Output: data/processed/pjme_processed.csv   one row per hour, all features + split label
        data/processed/scaler.json          mean/std of the TRAIN load
        data/processed/data_report.json     every number quoted in docs/dataset_documentation.md
"""
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
OUT = ROOT / "data" / "processed"


def load_raw() -> pd.DataFrame:
    d = CFG["dataset"]
    df = pd.read_csv(ROOT / d["raw_path"], parse_dates=[d["timestamp_col"]])
    return df.rename(columns={d["timestamp_col"]: "timestamp", d["target_col"]: "load_mw"})


def clean(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    report["raw_rows"] = int(len(df))
    report["raw_sorted"] = bool(df["timestamp"].is_monotonic_increasing)
    df = df.sort_values("timestamp")                       # step 1: the raw file is NOT in time order

    # step 2: duplicated timestamps = the repeated 01:00-02:00 hour when clocks go back (DST end)
    report["duplicate_timestamps"] = int(df["timestamp"].duplicated().sum())
    df = df.groupby("timestamp", as_index=False)["load_mw"].mean()

    # step 3: regular hourly grid; missing hours = the skipped hour when clocks go forward (DST start)
    grid = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(grid)
    df.index.name = "timestamp"
    df["imputed"] = df["load_mw"].isna().astype(int)
    report["missing_hours_imputed"] = int(df["imputed"].sum())
    # each gap is a single hour -> linear interpolation between neighbours uses only the
    # hour before and the hour after, so it cannot leak information across split borders
    df["load_mw"] = df["load_mw"].interpolate(method="linear", limit=3)
    assert df["load_mw"].isna().sum() == 0
    report["clean_rows"] = int(len(df))
    report["start"], report["end"] = str(df.index.min()), str(df.index.max())
    return df.reset_index()


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    # raw integer calendar fields (used for one-hot encoding in the representation study)
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.day_of_week         # 0 = Monday
    df["month"] = ts.dt.month
    df["doy"] = ts.dt.dayofyear
    # cyclical encoding: hour 23 and hour 0 end up close together, unlike raw integers
    for col, period in [("hour", 24), ("dow", 7), ("month", 12), ("doy", 365.25)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    years = range(ts.dt.year.min(), ts.dt.year.max() + 1)
    hol = holidays.country_holidays(CFG["features"]["holiday_country"], years=years, observed=True)
    df["is_holiday"] = ts.dt.date.map(lambda d: int(d in hol))
    df["holiday_name"] = ts.dt.date.map(lambda d: hol.get(d, ""))
    return df


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    s = CFG["split"]
    train_end, val_end = pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_end"])
    df["split"] = np.where(df["timestamp"] <= train_end, "train",
                           np.where(df["timestamp"] <= val_end, "val", "test"))
    return df


def fit_scaler(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    train = df.loc[df["split"] == "train", "load_mw"]          # TRAIN ONLY
    mean, std = float(train.mean()), float(train.std())
    json.dump({"mean": mean, "std": std, "fitted_on": "train",
               "train_end": CFG["split"]["train_end"]}, open(OUT / "scaler.json", "w"), indent=2)
    df["load_scaled"] = (df["load_mw"] - mean) / std
    # 3-sigma check (Dong et al., 2025, Sec. 6.1) - FLAGGED, not removed: extreme hours are
    # real events (heatwaves, Hurricane Sandy 2012-10-29/30) and are exactly what we must forecast
    df["flag_3sigma"] = (df["load_scaled"].abs() > 3).astype(int)
    report["scaler"] = {"mean": round(mean, 2), "std": round(std, 2)}
    report["flag_3sigma_hours"] = int(df["flag_3sigma"].sum())
    return df


def summarise(df: pd.DataFrame, report: dict) -> None:
    rep = {}
    for name, g in df.groupby("split", sort=False):
        rep[name] = {"start": str(g["timestamp"].min()), "end": str(g["timestamp"].max()),
                     "hours": int(len(g)), "share_pct": round(100 * len(g) / len(df), 1),
                     "holiday_days": int(g.loc[g["is_holiday"] == 1, "timestamp"].dt.date.nunique()),
                     "load_mean": round(float(g["load_mw"].mean()), 1),
                     "load_min": float(g["load_mw"].min()), "load_max": float(g["load_mw"].max())}
    report["splits"] = rep
    low = df.loc[df["load_mw"].idxmin()]
    high = df.loc[df["load_mw"].idxmax()]
    report["record_low"] = {"timestamp": str(low["timestamp"]), "load_mw": float(low["load_mw"])}
    report["record_high"] = {"timestamp": str(high["timestamp"]), "load_mw": float(high["load_mw"])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    df = clean(load_raw(), report)
    df = add_calendar(df)
    df = assign_split(df)
    df = fit_scaler(df, report)
    summarise(df, report)
    df.to_csv(OUT / "pjme_processed.csv", index=False)
    json.dump(report, open(OUT / "data_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
