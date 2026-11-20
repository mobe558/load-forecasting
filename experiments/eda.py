"""Exploratory figures for the dataset documentation.  Usage: python experiments/eda.py
Figures 2 and 3 use TRAIN data only, so no design decision is based on the test set."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.windows import load_processed  # noqa: E402

OUT = Path(__file__).parent / "eda"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.titlelocation": "left"})
df = load_processed()

# --- Figure 1: whole series with the chronological split -----------------------------
daily = df.set_index("timestamp")["load_mw"].resample("D").mean() / 1000
fig, ax = plt.subplots(figsize=(11, 3.6))
spans = {"train": ("2002-01-01", "2014-12-31"), "val": ("2015-01-01", "2016-12-31"),
         "test": ("2017-01-01", "2018-08-03")}
shade = {"train": "#ffffff00", "val": "#f0efec", "test": "#e1e0d9"}
for name, (a, b) in spans.items():
    ax.axvspan(np.datetime64(a), np.datetime64(b), color=shade[name], zorder=0, lw=0)
    mid = np.datetime64(a) + (np.datetime64(b) - np.datetime64(a)) / 2
    ax.text(mid, 56, {"train": "Train 2002-2014", "val": "Validation\n2015-16",
                      "test": "Test\n2017-18"}[name], ha="center", va="top", color=INK2, fontsize=9)
ax.plot(daily.index, daily.values, color=BLUE, lw=1.0)
ax.annotate("Hurricane Sandy\n(record low)", xy=(np.datetime64("2012-10-30"), daily["2012-10-30"]),
            xytext=(np.datetime64("2010-06-01"), 19), color=INK2, fontsize=8,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(15, 57)
ax.set_ylabel("Daily mean load (GW)")
ax.set_title("PJM East hourly load, 2002-2018, with the chronological split")
fig.tight_layout(); fig.savefig(OUT / "fig1_series_split.png", dpi=150); plt.close(fig)

# --- Figure 2: average daily profile by day type (train only) ------------------------
tr = df[df["split"] == "train"].copy()
tr["day_type"] = np.where(tr["is_holiday"] == 1, "Holiday",
                          np.where(tr["is_weekend"] == 1, "Weekend", "Weekday"))
prof = tr.groupby(["day_type", "hour"])["load_mw"].mean().unstack(0) / 1000
fig, ax = plt.subplots(figsize=(7.5, 4))
for col, c in [("Weekday", BLUE), ("Weekend", ORANGE), ("Holiday", AQUA)]:
    ax.plot(prof.index, prof[col], color=c, lw=2, label=col)
    ax.text(23.3, prof[col].iloc[-1], col, color=INK2, va="center", fontsize=9)
ax.set_xlim(0, 25.5); ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean load (GW)")
ax.set_title("Calendar matters: average daily profile by day type (train set)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig2_profile_daytype.png", dpi=150); plt.close(fig)

# --- Figure 3: autocorrelation (train only) -> justifies the 168-hour lookback ------
x = tr["load_mw"].to_numpy(); x = x - x.mean()
lags = np.arange(1, 24 * 15 + 1)
acf = np.array([np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in lags])
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(lags, acf, color=BLUE, lw=1.5)
for k, lab in [(24, "24 h (1 day)"), (168, "168 h (1 week)"), (336, "336 h (2 weeks)")]:
    ax.axvline(k, color=AXIS, lw=0.8, ls="--")
    ax.text(k + 3, 0.98, lab, color=INK2, fontsize=8, va="top")
ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Autocorrelation")
ax.set_ylim(min(acf.min(), 0) - 0.05, 1.02); ax.axhline(0, color=AXIS, lw=0.8)
ax.set_title("Load repeats every day and every week (train set)")
fig.tight_layout(); fig.savefig(OUT / "fig3_autocorrelation.png", dpi=150); plt.close(fig)
print("acf@24", round(acf[23], 3), "acf@168", round(acf[167], 3), "acf@12", round(acf[11], 3))
print("weekday peak", round(prof['Weekday'].max(), 2), "weekend peak", round(prof['Weekend'].max(), 2),
      "holiday peak", round(prof['Holiday'].max(), 2))
