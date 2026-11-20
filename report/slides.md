<!-- title -->
# Multi-horizon Electricity Load Forecasting

**Day-ahead forecasting of the PJM East grid with LSTM, Transformer and TFT**

Deep Learning AI403 · Final Project #13 · Pharos University in Alexandria · Fall 2026

**Team forecast_AI**

Mohab Mohamed Moustafa Mahmoud Mohamed Aboklela · 202303375<br>Rosana Zarif Sadek Semean · 202303306<br>Yousef Mohamed Ragab Abdo AboEita · 202302137<br>Avronia Magdy Tawfik Boutros · 202303908<br>Sandra Samer Samir Fargallah · 202303334

---

## 1 · Problem and motivation

<div class="cols" markdown="1">
<div markdown="1">

- Every day at **00:00**, forecast the **24 hourly loads** of PJM East (≈ 32 GW average, USA)
- **Input:** last 168 h of load + calendar (hour, weekday, season, holidays); calendar of the next 24 h is known
- **Output:** 24 loads in one forward pass (direct multi-horizon)
- **Why it matters:** 1 % less error ≈ 320 MW every hour — a whole gas turbine of reserves
- **Metrics:** MAE (primary), RMSE, MAPE, MAE per horizon; 3 seeds + bootstrap CI

</div>
<div markdown="1">

| # | Hypothesis |
|---|---|
| H1 | Best deep model beats the baselines |
| H2 | Calendar features help, most on weekends/holidays |
| H3 | Transformer beats LSTM at late horizons (h13–24) |
| H4 | Pretraining on other regions beats scratch |

</div>
</div>

---

## 2 · Data: PJM East hourly load, 2002 → 2018

{{fig:experiments/eda/fig1_series_split.png|145k hourly rows · chronological split: train 2002–2014 · validation 2015–16 · test 2017–Aug 2018 (579 day-ahead forecasts)}}

- Train-only z-score scaler · DST duplicates/gaps fixed · outliers kept (heatwaves, Hurricane Sandy are real)
- A window belongs to a split only if **all 24 target hours** are in it — checked automatically by `check_split.py`

---

## 2 · Method: one pipeline, five model families

<div class="cols" markdown="1">
<div markdown="1">

- **Baselines:** seasonal naive (last week / yesterday), **LightGBM** with lag features
- **Classic:** LSTM encoder–decoder (40k params)
- **Modern:** **patch Transformer** — 7 daily patches, cross-attention from each target hour into past days; **attention written by hand** (187k params)
- **Advanced:** **Temporal Fusion Transformer** — variable selection, gated residual networks, interpretable attention (44k params)
- **Transfer:** Transformer pretrained on **10 other PJM regions**, fine-tuned

</div>
<div markdown="1">

**Same protocol for every deep model**

- AdamW · L1 loss · batch 256 · ≤ 60 epochs · early stopping on validation
- **RevIN**: each window normalised by its own mean/SD (hand-written)
- Same LR grid {3e-4, 1e-3, 3e-3} for every model
- 3 seeds per final model · 49 runs in total, CPU only

</div>
</div>

---

## 3 · Results on the untouched test set

{{table:table4_main_results}}

**Transformer is best: 1,389 ± 12 MW (MAPE 4.28 %)** — 60 % better than "same hour last week", 3.5 % better than LightGBM

---

## 3 · Where in the day do the models err?

<div class="cols" markdown="1">
<div markdown="1">

{{fig:experiments/figures/per_horizon_mae.png|Test MAE per forecast hour}}

</div>
<div markdown="1">

- Error peaks at the **evening peak (18:00)**, not at hour 24
- Transformer gains most in the **early hours** (h1–12: 827 vs 959 MW for LSTM)
- In the second half of the day all models ≈ 1,950 MW: limited by missing weather, not by architecture
- Summer is the hardest season for every model

</div>
</div>

---

## 3 · Hypothesis verdicts

{{table:table_hypotheses}}

- **H1 supported** — Transformer beats LightGBM, CI excludes 0
- **H2 inconclusive** — direction as predicted (weekend/holiday effect 4× weekday), CI includes 0
- **H3 inconclusive** — the Transformer's advantage is in the *early* hours
- **H4 supported with little data** — pretraining: −12 % error with 2 years of data, no gain with 13 years

---

## 3 · Hyperparameter and data-strategy studies (validation)

<div class="cols" markdown="1">
<div markdown="1">

**Same LR grid for every model** (best val MAE in 30 epochs, MW)

| Model | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|
| Transformer | 1,456 | **1,265** | 1,269 |
| LSTM | 1,466 | 1,351 | **1,275** |
| TFT | 1,432 | 1,313 | **1,249** |

Transformer: d = 32 / 64 / 128 → 1,368 / 1,265 / 1,213 · dropout 0 / 0.1 / 0.3 → **1,176** / 1,265 / 1,512

</div>
<div markdown="1">

**Representation (Transformer, 3 seeds)**

| Variant | Val MAE | Test MAE |
|---|---|---|
| cyclical + RevIN (final) | 1,205 | 1,389 |
| no RevIN | 1,331 | 1,520 |
| **one-hot calendar** | **1,182** | **1,340** |

- **One-hot wins:** 4.13 % MAPE, 99 MW better than LightGBM
- Lookback 72 h / 336 h: no better than 168 h

</div>
</div>

---

## 4 · Ablation study

{{table:table5_ablation}}

- **RevIN is the most important component (+9.5 % error without it)** — distribution shift between 2002–14 and 2017–18
- Calendar features: +31 MW, mostly on weekends and holidays
- Removing attention (Transformer cross-attention, TFT attention layer): within seed noise

---

## 4 · Error analysis: where and why the models fail

<div class="cols" markdown="1">
<div markdown="1">

{{fig:experiments/figures/worst_days.png|The four worst test days}}

</div>
<div markdown="1">

- **8 of the 10 worst days**: daily load jumped > 2.5 GW vs the day before — mostly **a heat spell breaking** (over-forecast by up to 11 GW at the peak)
- **First day of a heatwave**: peaks under-forecast (−1.7 GW on the hottest 10 % of days)
- Holidays: one bit cannot separate Veterans Day from Christmas
- **Cause:** tomorrow's temperature is not in the input — LightGBM fails on the same days

</div>
</div>

---

## 5 · Conclusion and future work

- A leakage-free, reproducible day-ahead forecaster: **4.28 % MAPE** with a hand-written patch Transformer, **4.13 %** with a one-hot calendar
- Deep learning beats a strong LightGBM baseline — **significantly but modestly**
- **Inductive bias > size:** daily patches, per-window normalisation and known-future calendar inputs matter most
- Pretraining is a **data-efficiency** tool: big gain only when target data is scarce
- Remaining error is **weather**, not model capacity

**Next:** temperature forecasts as known future inputs · quantile (probabilistic) forecasts · holiday-type encoding · Transformer + LightGBM ensemble

---

<!-- title -->
# Live demo

`python demo/app.py` → http://127.0.0.1:7860

**Tab 1 – Replay a test day:** heatwave 2017-07-19 · Christmas 2017-12-25 · Thanksgiving 2017-11-23

**Tab 2 – Upload your own CSV:** `timestamp, load_mw`, ≥ 168 hourly rows → next 24 hours

Thank you — questions?
