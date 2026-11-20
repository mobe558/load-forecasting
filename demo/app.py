"""Live demo (Core Req. 12): Gradio app that takes REAL input and returns the 24-hour forecast.

  python demo/app.py          -> open http://127.0.0.1:7860

Two ways to give input:
  1. "Replay a test day": pick any day of the held-out test period (2017-01-01 ... 2018-08-02). The app
     takes the real 168 hours before it from the PJME data, forecasts the day with every trained model,
     and shows the actual load next to the forecasts (with the error of each model).
  2. "Upload your own CSV": a file with columns `timestamp, load_mw` and at least 168 hourly rows.
     The app forecasts the 24 hours after the last timestamp.
No external AI API is used: every forecast comes from the models trained in this repository.
"""
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import L, future_timestamps, load_all  # noqa: E402

DATA = pd.read_csv(ROOT / "data" / "processed" / "pjme_processed.csv", usecols=["timestamp", "load_mw"],
                   parse_dates=["timestamp"]).set_index("timestamp")["load_mw"]
MODELS = load_all()
COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#52514e", "#898781"]


def forecast_all(hist, hist_ts, fut_ts):
    return {name: np.asarray(m(hist, hist_ts, fut_ts), dtype=float) for name, m in MODELS.items()}


def make_outputs(fut_ts, preds, actual=None, title=""):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if actual is not None:
        ax.plot(fut_ts, actual / 1000, color="#0b0b0b", lw=2.6, label="Actual")
    for (name, p), c in zip(preds.items(), COLORS):
        ax.plot(fut_ts, p / 1000, color=c, lw=1.8, ls="--" if name in ("Seasonal naive", "LightGBM") else "-",
                label=name)
    ax.set_ylabel("Load (GW)"), ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=.3), ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(), fig.tight_layout()
    table = pd.DataFrame({"hour": [t.strftime("%Y-%m-%d %H:00") for t in fut_ts],
                          **{k: np.round(v).astype(int) for k, v in preds.items()}})
    if actual is not None:
        table.insert(1, "Actual", np.round(actual).astype(int))
        scores = pd.DataFrame([{"Model": k, "MAE (MW)": round(float(np.abs(v - actual).mean())),
                                "MAPE (%)": round(float(100 * np.mean(np.abs(v - actual) / actual)), 2),
                                "Peak error (MW)": round(float(v.max() - actual.max()))}
                               for k, v in preds.items()]).sort_values("MAE (MW)")
    else:
        scores = pd.DataFrame([{"Model": k, "Forecast peak (MW)": round(float(v.max())),
                                "Peak hour": fut_ts[int(v.argmax())].strftime("%H:00")} for k, v in preds.items()])
    return fig, scores, table


def replay(date_str):
    day = pd.Timestamp(date_str).normalize()
    if day < pd.Timestamp("2017-01-01") or day > pd.Timestamp("2018-08-02"):
        raise gr.Error("Pick a day of the test period: 2017-01-01 ... 2018-08-02")
    hist = DATA.loc[day - pd.Timedelta(hours=L): day - pd.Timedelta(hours=1)]
    fut_ts = pd.date_range(day, periods=24, freq="h")
    actual = DATA.loc[fut_ts].to_numpy()
    preds = forecast_all(hist.to_numpy(), hist.index, fut_ts)
    return make_outputs(fut_ts, preds, actual, f"Day-ahead forecast for {day:%A %d %b %Y}")


def upload(file):
    if file is None:
        raise gr.Error("Upload a CSV with columns timestamp, load_mw")
    df = pd.read_csv(file if isinstance(file, str) else file.name)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"timestamp", "load_mw"} <= set(df.columns):
        raise gr.Error("CSV needs the columns: timestamp, load_mw")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")["load_mw"].resample("h").mean().interpolate(limit=3)
    if len(df) < L or df.iloc[-L:].isna().any():
        raise gr.Error(f"Need at least {L} consecutive hourly values (got {len(df)})")
    hist = df.iloc[-L:]
    fut_ts = future_timestamps(hist.index[-1])
    preds = forecast_all(hist.to_numpy(), hist.index, fut_ts)
    return make_outputs(fut_ts, preds, None, f"Forecast for the 24 h after {hist.index[-1]:%d %b %Y %H:00}")


def example_csv():
    """Writes the last real week of the dataset as an example upload file."""
    p = ROOT / "demo" / "example_last_week.csv"
    DATA.iloc[-L:].rename("load_mw").reset_index().to_csv(p, index=False)
    return str(p)


with gr.Blocks(title="Day-ahead electricity load forecast") as app:
    gr.Markdown("# Day-ahead electricity load forecasting - PJM East\n"
                f"Models loaded: **{', '.join(MODELS)}**. Input: last 168 h of load + calendar -> output: next 24 h.")
    with gr.Tab("Replay a test day"):
        d = gr.Textbox(value="2017-07-19", label="Day to forecast (YYYY-MM-DD, test period)")
        b = gr.Button("Forecast", variant="primary")
        gr.Examples(["2017-07-19", "2017-12-25", "2018-01-02", "2017-11-23", "2018-05-28"], inputs=d,
                    label="Try: heatwave, Christmas, cold snap, Thanksgiving, Memorial Day")
        p1, s1, t1 = gr.Plot(), gr.Dataframe(label="Scores vs actual"), gr.Dataframe(label="Hourly forecast (MW)")
        b.click(replay, d, [p1, s1, t1])
    with gr.Tab("Upload your own CSV"):
        f = gr.File(label="CSV with columns timestamp, load_mw (>= 168 hourly rows)", file_types=[".csv"])
        b2 = gr.Button("Forecast next 24 h", variant="primary")
        ex = gr.Button("Download an example file")
        exf = gr.File(label="example")
        p2, s2, t2 = gr.Plot(), gr.Dataframe(label="Summary"), gr.Dataframe(label="Hourly forecast (MW)")
        b2.click(upload, f, [p2, s2, t2])
        ex.click(example_csv, None, exf)

if __name__ == "__main__":
    app.launch()
