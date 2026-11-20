"""Error analysis (Core Req. 10): where and why do the models fail?

  python experiments/error_analysis.py [--model transformer]

Writes experiments/tables/table6_error_analysis.md, worst_days.md and figures in experiments/figures/:
  * error by day type, month and load level (peak vs normal)
  * the worst test days with actual vs forecast curves
  * a categorised failure table: error type, count, example, likely cause
Uses the seed-averaged forecast of each model (ensemble of the 3 seeds) on the test set only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import FIG, LABEL, TAB, load_run, write  # noqa: E402


def seed_mean(name):
    s = load_run(name)
    return s[0]["origin"], s[0]["y"], np.mean([x["p"] for x in s], axis=0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="transformer")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    proc = pd.read_csv(ROOT / "data" / "processed" / "pjme_processed.csv",
                       usecols=["timestamp", "is_holiday", "holiday_name", "imputed"], parse_dates=["timestamp"])
    proc["date"] = proc["timestamp"].dt.normalize()
    daily = proc.groupby("date").agg(is_holiday=("is_holiday", "max"), holiday=("holiday_name", "first"),
                                     imputed=("imputed", "max"))
    org, y, p = seed_mean(a.model)
    _, _, p_lgb = seed_mean("lightgbm")
    err = np.abs(y - p)
    d = pd.DataFrame({"date": org.normalize(), "mae": err.mean(1), "bias": (p - y).mean(1),
                      "peak_actual": y.max(1), "peak_err": p.max(1) - y.max(1), "mae_lgb": np.abs(y - p_lgb).mean(1)})
    d = d.join(daily, on="date")
    d["dow"] = d["date"].dt.day_name()
    # the day BEFORE/AFTER a holiday behaves differently too
    hol_dates = set(daily.index[daily["is_holiday"] == 1])
    d["near_holiday"] = d["date"].apply(lambda x: (x - pd.Timedelta("1D") in hol_dates) or
                                        (x + pd.Timedelta("1D") in hol_dates)).astype(int)
    # change of daily mean load vs the previous day: proxy for a sudden weather change
    d["jump"] = np.r_[np.nan, np.diff(y.mean(1))]
    p95 = np.quantile(y.max(1), 0.9)
    mu, sd = d["mae"].mean(), d["mae"].std()

    # ---- categorised failures (a day can belong to several categories; first match is used)
    def category(r):
        if r.is_holiday == 1:
            return "Public holiday"
        if r.near_holiday == 1:
            return "Day before/after a holiday"
        if r.peak_actual >= p95:
            return "Extreme peak day (top 10% peak load)"
        if abs(r.jump) > 2500 if not np.isnan(r.jump) else False:
            return "Sudden level change vs previous day (weather front)"
        if r.date.month in (3, 11) and r.date.day_of_week == 6 and r.date.day <= 14 and (
                (r.date.month == 3 and 8 <= r.date.day <= 14) or (r.date.month == 11 and r.date.day <= 7)):
            return "Daylight-saving change"
        return "Normal day"

    d["category"] = d.apply(category, axis=1)
    bad = d["mae"] > mu + sd                               # "large error" days
    cause = {"Public holiday": "Only ~11 holidays/year in training; holiday flag is one bit for very different "
                               "holidays (Christmas vs Columbus Day)",
             "Day before/after a holiday": "Bridge days are not encoded; behaviour is between weekday and holiday",
             "Extreme peak day (top 10% peak load)": "Heatwave/cold snap: no temperature input, the model only sees "
                                                     "last week's level and under-forecasts the peak",
             "Sudden level change vs previous day (weather front)": "Weather change between the forecast issue time and "
                                                                    "the target day is not observable from load alone",
             "Daylight-saving change": "Shifted daily profile around the clock change",
             "Normal day": "Residual noise / unmodelled factors"}
    rows = []
    for c, g in d.groupby("category"):
        worst = g.sort_values("mae", ascending=False).iloc[0]
        rows.append({"Error type": c, "Days in test": len(g), "Large-error days (MAE > mean+1SD)": int(bad[g.index].sum()),
                     "Mean MAE (MW)": f"{g['mae'].mean():,.0f}", "Mean bias (MW)": f"{g['bias'].mean():+,.0f}",
                     "Example (worst day)": f"{worst.date.date()} ({worst.dow[:3]}{', ' + worst.holiday if isinstance(worst.holiday, str) else ''}): MAE {worst.mae:,.0f}",
                     "Likely cause": cause[c]})
    t6 = pd.DataFrame(rows).sort_values("Mean MAE (MW)", key=lambda s: s.str.replace(",", "").astype(float),
                                        ascending=False)
    write(t6, "table6_error_analysis", f"Table 6 - Error analysis ({LABEL.get(a.model, a.model)}, seed-averaged, test set)")

    worst = d.sort_values("mae", ascending=False).head(10)
    w = worst[["date", "dow", "holiday", "category", "mae", "bias", "peak_actual", "peak_err", "mae_lgb"]].copy()
    w["date"] = w["date"].dt.date
    w["holiday"] = w["holiday"].fillna("")
    for c in ["mae", "bias", "peak_actual", "peak_err", "mae_lgb"]:
        w[c] = w[c].map(lambda v: f"{v:,.0f}")
    w.columns = ["Date", "Weekday", "Holiday", "Category", "MAE", "Bias (+ = over)", "Actual peak", "Peak error",
                 "LightGBM MAE"]
    write(w, "worst_days", "The 10 worst test days")

    # ---- figures
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)
    for ax, (_, r) in zip(axes.flat, worst.head(4).iterrows()):
        i = np.where(org.normalize() == r.date)[0][0]
        ax.plot(range(24), y[i] / 1000, color="#0b0b0b", lw=2, label="actual")
        ax.plot(range(24), p[i] / 1000, color="#2a78d6", lw=2, label=LABEL.get(a.model, a.model))
        ax.plot(range(24), p_lgb[i] / 1000, color="#898781", lw=1.2, ls="--", label="LightGBM")
        ax.set_title(f"{r.date.date()} {r.dow[:3]} - {r.category[:32]}", fontsize=9, loc="left")
        ax.set_xticks(range(0, 24, 6)), ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7), axes[1, 0].set_xlabel("Hour"), axes[1, 1].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Load (GW)"), axes[1, 0].set_ylabel("Load (GW)")
    fig.suptitle("Worst test days: actual vs forecast", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(), fig.savefig(FIG / "worst_days.png", dpi=150), plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    m = d.groupby(d["date"].dt.month)["mae"].mean()
    axes[0].bar(m.index, m.values, color="#2a78d6", width=0.7)
    axes[0].set_xticks(range(1, 13)), axes[0].set_xlabel("Month"), axes[0].set_ylabel("Mean daily MAE (MW)")
    axes[0].set_title("Error by month", loc="left", fontweight="bold", fontsize=10)
    axes[1].scatter(d["peak_actual"] / 1000, d["peak_err"] / 1000, s=10, color="#2a78d6", alpha=.6)
    axes[1].axhline(0, color="#c3c2b7", lw=1)
    axes[1].set_xlabel("Actual daily peak (GW)"), axes[1].set_ylabel("Forecast peak - actual peak (GW)")
    axes[1].set_title("Peaks are under-forecast on the hottest days", loc="left", fontweight="bold", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False), ax.grid(alpha=.3)
    fig.tight_layout(), fig.savefig(FIG / "error_by_month_and_peak.png", dpi=150), plt.close(fig)
    corr = np.corrcoef(d["peak_actual"], d["peak_err"])[0, 1]
    print(f"\ncorr(actual peak, peak error) = {corr:.2f}; top-10% peak days mean peak error "
          f"{d.loc[d.peak_actual >= p95, 'peak_err'].mean():+,.0f} MW")
    d.to_csv(TAB / "daily_errors.csv", index=False)


if __name__ == "__main__":
    main()
