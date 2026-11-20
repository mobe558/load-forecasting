"""Shared evaluation for ALL models (Core Req. 6, 9, 11 + hypothesis verdicts).

  python src/evaluate.py

Reads results/runs/<name>/seed*/ and writes
  experiments/tables/*.md|csv   main results, per-class (day type / season), per-horizon, HP study,
                                data strategy, pretrained-vs-scratch, ablation, cost, hypotheses
  experiments/figures/*.png     per-horizon MAE, model comparison, learning curves
Metrics are computed in MW on the untouched test set (2017-01-01 -> 2018-08-02, 579 day-ahead forecasts).
Uncertainty: mean +/- SD over 3 seeds, and a 95% bootstrap CI over test days.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import bootstrap_ci  # noqa: E402

RUNS = ROOT / "results" / "runs"
TAB = ROOT / "experiments" / "tables"
FIG = ROOT / "experiments" / "figures"

LABEL = {"naive_week": "Seasonal naive (last week)", "naive_day": "Seasonal naive (yesterday)",
         "lightgbm": "LightGBM (baseline)", "lstm": "LSTM enc-dec (classic)",
         "transformer": "Transformer (modern, hand-written attn)", "tft": "TFT (advanced)",
         "transformer_finetune": "Transformer pretrained+fine-tuned"}
MAIN = list(LABEL)


# ------------------------------------------------------------------------------------------ loading
def load_run(name, split="test"):
    seeds = []
    for d in sorted((RUNS / name).glob("seed*")):
        f = d / f"preds_{split}.npz"
        if (d / "metrics.json").exists() and f.exists():
            z = np.load(f)
            seeds.append({"seed": d.name, "origin": pd.to_datetime(z["origin"]), "y": z["y_true"],
                          "p": z["y_pred"], "cost": json.load(open(d / "cost.json")),
                          "metrics": json.load(open(d / "metrics.json"))})
    return seeds


def day_info():
    df = pd.read_csv(ROOT / "data" / "processed" / "pjme_processed.csv", usecols=["timestamp", "is_holiday"],
                     parse_dates=["timestamp"])
    df = df[df["timestamp"].dt.hour == 0].set_index("timestamp")
    return df["is_holiday"]


def day_type(origins, hol):
    h = hol.reindex(origins).fillna(0).to_numpy()
    wk = origins.day_of_week >= 5
    return np.where(h == 1, "holiday", np.where(wk, "weekend", "weekday"))


def season(origins):
    m = origins.month
    return np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])],
                     ["winter", "spring", "summer"], "autumn")


def summarise(name, split="test"):
    seeds = load_run(name, split)
    if not seeds:
        return None
    mae = np.array([np.abs(s["y"] - s["p"]).mean() for s in seeds])
    rmse = np.array([np.sqrt(((s["y"] - s["p"]) ** 2).mean()) for s in seeds])
    mape = np.array([100 * (np.abs(s["y"] - s["p"]) / s["y"]).mean() for s in seeds])
    per_day = np.mean([np.abs(s["y"] - s["p"]).mean(1) for s in seeds], axis=0)   # seed-averaged
    lo, hi = bootstrap_ci(per_day)
    per_h = np.mean([np.abs(s["y"] - s["p"]).mean(0) for s in seeds], axis=0)
    return {"name": name, "n_seeds": len(seeds), "MAE": mae.mean(), "MAE_sd": mae.std(ddof=1) if len(mae) > 1 else 0.0,
            "RMSE": rmse.mean(), "RMSE_sd": rmse.std(ddof=1) if len(rmse) > 1 else 0.0,
            "MAPE": mape.mean(), "MAPE_sd": mape.std(ddof=1) if len(mape) > 1 else 0.0,
            "CI_lo": lo, "CI_hi": hi, "per_day": per_day, "per_h": per_h, "origin": seeds[0]["origin"],
            "seeds": seeds}


def fmt(m, sd, d=0):
    return f"{m:,.{d}f} ± {sd:,.{d}f}"


def write(df, stem, title):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{stem}.csv", index=False)
    with open(TAB / f"{stem}.md", "w") as f:
        f.write(f"### {title}\n\n" + df.to_markdown(index=False) + "\n")
    print(f"\n### {title}\n" + df.to_markdown(index=False))


def paired_diff(a, b, mask=None, n_boot=2000, seed=0):
    """Bootstrap CI of mean(daily MAE of a - daily MAE of b) over test days; negative = a better."""
    d = a["per_day"] - b["per_day"]
    if mask is not None:
        d = d[mask]
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return d.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def per_h_diff(a, b, hours):
    da = np.mean([np.abs(s["y"] - s["p"])[:, hours].mean(1) for s in a["seeds"]], axis=0)
    db = np.mean([np.abs(s["y"] - s["p"])[:, hours].mean(1) for s in b["seeds"]], axis=0)
    d = da - db
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return d.mean(), np.quantile(bs, 0.025), np.quantile(bs, 0.975)


# ------------------------------------------------------------------------------------------ tables
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    hol = day_info()
    S = {n: summarise(n) for n in [p.name for p in RUNS.iterdir() if p.is_dir()]}
    S = {k: v for k, v in S.items() if v}
    V = {n: summarise(n, "val") for n in [p.name for p in RUNS.iterdir() if p.is_dir()]}
    V = {k: v for k, v in V.items() if v}

    # ---- Table 4: main results
    rows = []
    for n in MAIN:
        if n in S:
            s = S[n]
            rows.append({"Model": LABEL[n], "Seeds": s["n_seeds"], "Test MAE (MW)": fmt(s["MAE"], s["MAE_sd"]),
                         "95% CI (MW)": f"[{s['CI_lo']:,.0f}, {s['CI_hi']:,.0f}]",
                         "RMSE (MW)": fmt(s["RMSE"], s["RMSE_sd"]), "MAPE (%)": fmt(s["MAPE"], s["MAPE_sd"], 2),
                         "Val MAE (MW)": f"{V[n]['MAE']:,.0f}" if n in V else ""})
    write(pd.DataFrame(rows), "table4_main_results", "Table 4 - Main results on the test set (mean ± SD over seeds)")

    # ---- per-class results: day type and season
    rows = []
    for n in MAIN:
        if n not in S:
            continue
        s = S[n]
        dt, se = day_type(s["origin"], hol), season(s["origin"])
        r = {"Model": LABEL[n]}
        for c in ["weekday", "weekend", "holiday"]:
            r[f"{c} (n={int((dt == c).sum())})"] = f"{s['per_day'][dt == c].mean():,.0f}"
        for c in ["winter", "spring", "summer", "autumn"]:
            r[f"{c} (n={int((se == c).sum())})"] = f"{s['per_day'][se == c].mean():,.0f}"
        rows.append(r)
    write(pd.DataFrame(rows), "table4b_per_class", "Per-class test MAE (MW): by day type and by season")

    # ---- per-horizon
    rows = []
    for n in MAIN:
        if n in S:
            ph = S[n]["per_h"]
            rows.append({"Model": LABEL[n], "h=1": f"{ph[0]:,.0f}", "h=6": f"{ph[5]:,.0f}", "h=12": f"{ph[11]:,.0f}",
                         "h=18": f"{ph[17]:,.0f}", "h=24": f"{ph[23]:,.0f}",
                         "mean h1-12": f"{ph[:12].mean():,.0f}", "mean h13-24": f"{ph[12:].mean():,.0f}"})
    write(pd.DataFrame(rows), "table4c_per_horizon", "Test MAE (MW) per forecast horizon (hour of the day ahead)")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = {"lightgbm": "#52514e", "lstm": "#eb6834", "transformer": "#2a78d6", "tft": "#1baf7a",
              "transformer_finetune": "#6250d6", "naive_day": "#898781"}
    for n, c in colors.items():
        if n in S:
            ax.plot(range(1, 25), S[n]["per_h"], color=c, lw=2, label=LABEL[n],
                    ls="--" if n in ("lightgbm", "naive_day") else "-")
    ax.set_xlabel("Forecast hour (00:00 = 1)"), ax.set_ylabel("Test MAE (MW)")
    ax.set_title("Error across the 24-hour horizon", loc="left", fontweight="bold")
    ax.grid(alpha=.3), ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(), fig.savefig(FIG / "per_horizon_mae.png", dpi=150), plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ns = [n for n in MAIN if n in S]
    ax.barh([LABEL[n] for n in ns][::-1], [S[n]["MAE"] for n in ns][::-1],
            xerr=[S[n]["MAE_sd"] for n in ns][::-1], color="#2a78d6", height=0.6)
    for i, n in enumerate(ns[::-1]):
        ax.text(S[n]["MAE"] + 40, i, f"{S[n]['MAE']:,.0f}", va="center", fontsize=8, color="#52514e")
    ax.set_xlabel("Test MAE (MW), mean ± SD over seeds")
    ax.set_title("Model comparison on the test set", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(), fig.savefig(FIG / "model_comparison.png", dpi=150), plt.close(fig)

    # ---- HP study (validation only)
    hp = [("transformer", "lr", "1e-3 (default)"), ("hp_transformer_lr3e-4", "lr", "3e-4"),
          ("hp_transformer_lr3e-3", "lr", "3e-3"), ("hp_transformer_d32", "d_model", "32"),
          ("hp_transformer_d128", "d_model", "128"), ("hp_transformer_drop0", "dropout", "0.0"),
          ("hp_transformer_drop3", "dropout", "0.3"),
          ("lstm", "lr", "1e-3 (default)"), ("hp_lstm_lr3e-4", "lr", "3e-4"), ("hp_lstm_lr3e-3", "lr", "3e-3"),
          ("tft", "lr", "1e-3 (default)"), ("hp_tft_lr3e-4", "lr", "3e-4"), ("hp_tft_lr3e-3", "lr", "3e-3")]
    rows = []
    for n, p, v in hp:
        h = RUNS / n / "seed0" / "history.csv"
        c = RUNS / n / "seed0" / "cost.json"
        if h.exists() and c.exists():
            hist = pd.read_csv(h).head(30)              # same 30-epoch screening budget for every setting
            rows.append({"Model": n.replace("hp_", "").split("_")[0], "Hyperparameter": p, "Value": v,
                         "Best val MAE in 30 epochs (MW)": f"{hist['val_mae_mw'].min():,.0f}",
                         "Best epoch": int(hist.loc[hist['val_mae_mw'].idxmin(), 'epoch']),
                         "Params": f"{json.load(open(c))['params']:,}"})
    write(pd.DataFrame(rows), "table_hp_study",
          "Hyperparameter study (validation set, seed 0, equal 30-epoch screening budget per setting)")

    # ---- data strategy
    ds = [("transformer", "cyclical sin/cos + RevIN, lookback 168 (final)"),
          ("ds_transformer_norevin", "global z-score only (no RevIN)"),
          ("ds_transformer_onehot", "one-hot calendar (45 features)"),
          ("ds_transformer_lb72", "lookback 72 h"), ("ds_transformer_lb336", "lookback 336 h")]
    rows = []
    for n, d in ds:
        v = summarise(n, "val")
        if v:
            t = S.get(n)
            rows.append({"Representation": d, "Seeds": v["n_seeds"], "Val MAE (MW)": fmt(v["MAE"], v["MAE_sd"]),
                         "Test MAE (MW)": fmt(t["MAE"], t["MAE_sd"]) if t else "not evaluated (tuning)"})
    write(pd.DataFrame(rows), "table_data_strategy", "Data-strategy study (Transformer)")

    # ---- pretrained vs scratch
    pt = [("transformer", "scratch", "full (2002-2014)"), ("transformer_finetune", "pretrained + fine-tuned", "full (2002-2014)"),
          ("transformer_pretrain", "pretrained only, zero-shot on PJME", "none (other regions)"),
          ("pt_scratch_2y", "scratch", "2 years (2013-2014)"), ("pt_finetune_2y", "pretrained + fine-tuned", "2 years (2013-2014)")]
    rows = []
    for n, init, data in pt:
        t = S.get(n)
        if t:
            rows.append({"Init": init, "PJME training data": data, "Seeds": t["n_seeds"],
                         "Test MAE (MW)": fmt(t["MAE"], t["MAE_sd"]), "MAPE (%)": fmt(t["MAPE"], t["MAPE_sd"], 2),
                         "Train time (s)": f"{np.mean([s['cost']['train_time_s'] for s in t['seeds']]):,.0f}"})
    write(pd.DataFrame(rows), "table_pretrained_vs_scratch", "Pretrained vs scratch (Transformer)")

    # ---- Table 5: ablation
    ab = [("transformer", "transformer", "- (full model)"),
          ("abl_transformer_nocal", "transformer", "calendar features (all 10)"),
          ("abl_transformer_noattn", "transformer", "decoder cross-attention"),
          ("ds_transformer_norevin", "transformer", "RevIN instance normalisation"),
          ("tft", "tft", "- (full model)"), ("abl_tft_noattn", "tft", "interpretable attention layer")]
    rows = []
    for n, base, comp in ab:
        t, b = S.get(n), S.get(base)
        if t and b:
            dt = day_type(t["origin"], hol)
            hd = t["per_day"][dt != "weekday"].mean() - b["per_day"][dt != "weekday"].mean()
            wd = t["per_day"][dt == "weekday"].mean() - b["per_day"][dt == "weekday"].mean()
            rows.append({"Model": base, "Component removed": comp, "Test MAE (MW)": fmt(t["MAE"], t["MAE_sd"]),
                         "Δ vs full (MW)": "" if n == base else f"{t['MAE'] - b['MAE']:+,.0f} ({100 * (t['MAE'] / b['MAE'] - 1):+.1f}%)",
                         "Δ weekdays": "" if n == base else f"{wd:+,.0f}",
                         "Δ weekends+holidays": "" if n == base else f"{hd:+,.0f}"})
    write(pd.DataFrame(rows), "table5_ablation", "Table 5 - Ablation study (test set, mean ± SD over 3 seeds)")

    # ---- Table 7: computational cost
    rows = []
    for n in MAIN + ["transformer_pretrain"]:
        if n in S:
            c = [s["cost"] for s in S[n]["seeds"]]
            hw = c[0]["hardware"]
            rows.append({"Model": LABEL.get(n, n), "Parameters": f"{c[0]['params']:,}" + (" trees" if n == "lightgbm" else ""),
                         "Size (MB)": f"{c[0]['model_size_mb']:.2f}",
                         "Train time (s)": fmt(np.mean([x["train_time_s"] for x in c]), np.std([x["train_time_s"] for x in c])),
                         "Epochs": fmt(np.mean([x.get("epochs", 0) for x in c]), np.std([x.get("epochs", 0) for x in c])) if n not in ("lightgbm", "naive_week", "naive_day") else "-",
                         "Inference (ms / day-ahead forecast, batched)": f"{np.mean([x['inference_ms_per_forecast'] for x in c]):.3f}",
                         "Latency (ms, single forecast)": f"{np.mean([x.get('inference_ms_single', np.nan) for x in c]):.1f}" if "inference_ms_single" in c[0] else "-",
                         "Peak RAM (MB)": f"{np.mean([x['peak_mem_mb'] for x in c]):.0f}",
                         "Hardware": f"{hw['machine']} CPU, {hw['cpu_count']} cores, {hw['torch_threads'] or '-'} thread(s)"})
    write(pd.DataFrame(rows), "table7_cost", "Table 7 - Computational cost")

    # ---- hypothesis verdicts
    rows = []
    dl = [n for n in ("lstm", "transformer", "tft", "transformer_finetune") if n in S]
    if dl and "lightgbm" in S:
        best = min(dl, key=lambda n: S[n]["MAE"])
        d, lo, hi = paired_diff(S[best], S["lightgbm"])
        rows.append({"Hypothesis": "H1 best DL model beats the baseline", "Comparison": f"{best} - lightgbm",
                     "Δ MAE (MW)": f"{d:+,.0f}", "95% CI": f"[{lo:+,.0f}, {hi:+,.0f}]",
                     "Verdict": "supported" if hi < 0 else ("rejected" if lo > 0 else "inconclusive")})
    if "abl_transformer_nocal" in S and "transformer" in S:
        d, lo, hi = paired_diff(S["abl_transformer_nocal"], S["transformer"])
        dt = day_type(S["transformer"]["origin"], hol)
        d2, lo2, hi2 = paired_diff(S["abl_transformer_nocal"], S["transformer"], dt != "weekday")
        d3, _, _ = paired_diff(S["abl_transformer_nocal"], S["transformer"], dt == "weekday")
        rows.append({"Hypothesis": "H2 calendar features reduce error (most on weekends/holidays)",
                     "Comparison": "no-calendar - full (all days / weekend+holiday / weekday)",
                     "Δ MAE (MW)": f"{d:+,.0f} / {d2:+,.0f} / {d3:+,.0f}", "95% CI": f"[{lo:+,.0f}, {hi:+,.0f}] (all days)",
                     "Verdict": ("supported" if lo > 0 and d2 > d3 else "partly supported" if lo > 0
                                 else "rejected" if hi < 0 else
                                 "inconclusive (direction as predicted)" if d > 0 and d2 > d3 else "inconclusive")})
    if "lstm" in S and "transformer" in S:
        d, lo, hi = per_h_diff(S["transformer"], S["lstm"], slice(12, 24))
        d1, _, _ = per_h_diff(S["transformer"], S["lstm"], slice(0, 12))
        rows.append({"Hypothesis": "H3 Transformer beats LSTM at later horizons (h13-24)",
                     "Comparison": "transformer - lstm (h13-24 / h1-12)", "Δ MAE (MW)": f"{d:+,.0f} / {d1:+,.0f}",
                     "95% CI": f"[{lo:+,.0f}, {hi:+,.0f}] (h13-24)",
                     "Verdict": "supported" if hi < 0 and d < d1 else ("partly supported" if hi < 0 else "rejected" if lo > 0 else "inconclusive")})
    for full, scr, lab in [("transformer_finetune", "transformer", "full data"), ("pt_finetune_2y", "pt_scratch_2y", "2-year data")]:
        if full in S and scr in S:
            d, lo, hi = paired_diff(S[full], S[scr])
            rows.append({"Hypothesis": f"H4 pretraining helps ({lab})", "Comparison": f"{full} - {scr}",
                         "Δ MAE (MW)": f"{d:+,.0f}", "95% CI": f"[{lo:+,.0f}, {hi:+,.0f}]",
                         "Verdict": "supported" if hi < 0 else ("rejected" if lo > 0 else "inconclusive")})
    write(pd.DataFrame(rows), "table_hypotheses", "Hypothesis verdicts (paired bootstrap over test days, negative Δ = first model better)")

    # ---- learning curves
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for n, c in [("lstm", "#eb6834"), ("transformer", "#2a78d6"), ("tft", "#1baf7a")]:
        f = RUNS / n / "seed0" / "history.csv"
        if f.exists():
            h = pd.read_csv(f)
            ax.plot(h["epoch"] * 8192 / 1000, h["val_mae_mw"], color=c, lw=1.6, label=LABEL[n])
    ax.axhline(V["lightgbm"]["MAE"] if "lightgbm" in V else np.nan, color="#52514e", ls="--", lw=1, label="LightGBM (val)")
    ax.set_ylim(top=2500), ax.set_xlabel("Training windows seen (thousands)"), ax.set_ylabel("Validation MAE (MW)")
    ax.set_title("Learning curves (seed 0)", loc="left", fontweight="bold"), ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(), fig.savefig(FIG / "learning_curves.png", dpi=150), plt.close(fig)

    json.dump({n: {"MAE": float(s["MAE"]), "MAE_sd": float(s["MAE_sd"]), "RMSE": float(s["RMSE"]),
                   "MAPE": float(s["MAPE"]), "CI": [float(s["CI_lo"]), float(s["CI_hi"])], "seeds": s["n_seeds"],
                   "per_h": [round(float(x), 1) for x in s["per_h"]]} for n, s in S.items()},
              open(TAB / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
