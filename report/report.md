<div class="title" markdown="1">

# Multi-horizon Electricity Load Forecasting

**Day-ahead forecasting of the PJM East grid with LSTM, Transformer and Temporal Fusion Transformer models**

Deep Learning – AI403 · Final Project (idea #13)<br>
Pharos University in Alexandria · Faculty of Computer Science & AI · Fall 2026

**Team:** forecast_AI

| Member | Student ID |
|---|---|
| Mohab Mohamed Moustafa Mahmoud Mohamed Aboklela | 202303375 |
| Rosana Zarif Sadek Semean | 202303306 |
| Yousef Mohamed Ragab Abdo AboEita | 202302137 |
| Avronia Magdy Tawfik Boutros | 202303908 |
| Sandra Samer Samir Fargallah | 202303334 |


Instructor: Dr. Sahar Ghanem · TAs: Esraa Habiba, Philopater Hany, Mohamed Ashraf

</div>

<div class="pb"></div>

## 1. Abstract

Grid operators commit generation a day ahead, so every megawatt of forecast error costs money in reserves or
imbalance penalties. We build a complete, reproducible deep-learning system that, every day at 00:00, forecasts
the **24 hourly loads of the PJM East region** (USA) from the previous 168 hours of load and calendar information.
On a strictly chronological split (train 2002–2014, validation 2015–2016, test 2017–Aug 2018, 579 day-ahead
forecasts) we compare two seasonal-naive rules and a LightGBM baseline with three deep models trained under one
protocol and three seeds each: an **LSTM encoder–decoder** (classic), a **patch Transformer with hand-written
attention** (modern) and a compact **Temporal Fusion Transformer** (advanced), plus a Transformer pretrained on ten
other PJM regions and fine-tuned on PJM East. The Transformer is the most accurate model: **test MAE 1,389 ± 12 MW
(MAPE 4.28 %)**, 51 MW better than LightGBM (paired bootstrap 95 % CI [−98, −4] MW) and 60 % better than the
"same hour last week" rule. Ablations show that **per-window instance normalisation (RevIN)** is the single most
important component (+9.5 % error without it), calendar features help most on weekends and holidays, and
cross-region pretraining does not help when 13 years of target data are available but cuts error by 12 % when only
two years are. Replacing the cyclical calendar encoding by one-hot indicators (selected on validation) improves the
Transformer further to **1,340 ± 10 MW (MAPE 4.13 %)**. Error analysis shows the remaining error is dominated by **weather-driven days** (heatwaves and the
day a heat spell breaks) that load history alone cannot anticipate.

## 2. Introduction

Short-term electricity load forecasting (STLF) — predicting load from one hour to a few days ahead — underpins unit
commitment, economic dispatch and day-ahead market bidding (Hong & Fan, 2016). Errors are asymmetric in practice:
under-forecasting forces expensive peaking plants or emergency imports, over-forecasting wastes committed capacity.
Load is strongly periodic (daily, weekly and yearly cycles), shaped by the calendar (weekends and holidays) and
driven by weather (air conditioning in summer, heating in winter).

Deep learning has become the dominant approach in the STLF literature (Dong et al., 2025), first with recurrent
networks (Kong et al., 2019) and more recently with attention-based models such as the Temporal Fusion Transformer
(Lim et al., 2021) and patch-based Transformers (Nie et al., 2023). At the same time, strong gradient-boosting
baselines are hard to beat on tabular-style forecasting problems, and recent work questions whether Transformers
really help for time series (Zeng et al., 2023). This project therefore puts the deep models through a fair,
hypothesis-driven comparison against well-tuned baselines, rather than reporting a single best number.

**Contributions.** (i) A leakage-free data pipeline with automatic split assertions; (ii) three deep architectures
of different families trained under the same budget with 3 seeds each, all written from the papers (scaled
dot-product attention, multi-head attention, patch embedding, positional encoding, RevIN, GRN/VSN and interpretable
attention are hand-written); (iii) a hyperparameter study, a data-representation study, a pretrained-vs-scratch study
and an ablation study; (iv) hypothesis verdicts with paired bootstrap confidence intervals; (v) an error analysis
with concrete failure days; and (vi) a live Gradio demo.

## 3. Problem Statement

**Task.** Day-ahead, multi-horizon point forecasting. At forecast origin *t* (00:00 of day *d*) predict the vector
ŷ = (ŷ<sub>t</sub>, …, ŷ<sub>t+23</sub>) of the 24 hourly loads of day *d* in MW.

**Input.** (a) the last 168 hourly loads (z-scored with training statistics); (b) 10 calendar features for those 168
hours; (c) the same 10 calendar features for the 24 target hours, which are known in advance. Calendar features:
sin/cos of hour, day-of-week, month and day-of-year, `is_weekend`, `is_holiday` (US federal, observed).
No weather data is used (Section 18).

**Output.** 24 loads produced in one forward pass (direct multi-output strategy), so errors cannot compound as in
recursive one-step forecasting.

**Metrics.** Primary: MAE in MW over all 24 horizons and test days. Secondary: RMSE (sensitive to large peak misses),
MAPE (scale-free, comparable across papers) and MAE per horizon *h* = 1…24. Uncertainty: mean ± SD over 3 seeds and a
95 % bootstrap CI over test days (Efron & Tibshirani, 1993).

**Hypotheses** (defined before running experiments, `docs/problem_definition.md`):

| # | Hypothesis | Test |
|---|---|---|
| H1 | The best deep model beats the baselines (seasonal naive, LightGBM). | Main comparison, paired bootstrap over test days |
| H2 | Calendar features reduce error, most of all on weekends and holidays. | Ablation: no calendar features |
| H3 | The Transformer keeps its accuracy better than the LSTM at later horizons (h = 13…24). | Per-horizon MAE |
| H4 | Pretraining on other regions + fine-tuning beats training from scratch. | Pretrained vs scratch, full and 2-year data |

## 4. Motivation

PJM Interconnection is the largest regional transmission organisation in the USA. A 1 % reduction of day-ahead MAE
at PJM East (mean load ≈ 32 GW) is ≈ 320 MW — the output of a mid-size gas turbine — every hour. Day-ahead forecasts are
also a canonical *multi-horizon* problem: hour 1 is almost a persistence problem, while hour 18 (the evening peak) must
be inferred from weekly patterns and the calendar. This makes the task a good test-bed for the course's central
question — when do recurrent, attention and gating mechanisms actually help? — on a problem small enough to train
3 seeds of every model on a laptop CPU.

## 5. Related Work

**Statistical and machine-learning STLF.** Seasonal naive and exponential smoothing models remain standard reference
points (Hyndman & Athanasopoulos, 2021). Gradient-boosted trees with lag features won several GEFCom load tracks and
are the strongest non-neural baselines (Hong & Fan, 2016; Ke et al., 2017).

**Recurrent networks.** LSTMs (Hochreiter & Schmidhuber, 1997) were the first deep models to clearly outperform
classical methods on load data (Kong et al., 2019). Sequence-to-sequence encoder–decoders (Sutskever et al., 2014)
extend them to multi-horizon outputs.

**Attention models.** The Transformer (Vaswani et al., 2017) replaces recurrence by self-attention. PatchTST
(Nie et al., 2023) groups the input series into patches, which shortens sequences and captures local shape; RevIN
(Kim et al., 2022) normalises each input window to counter distribution shift. The Temporal Fusion Transformer
(Lim et al., 2021) combines variable-selection networks, gated residual networks, an LSTM encoder–decoder and
interpretable multi-head attention, and was designed explicitly for multi-horizon forecasting with known future inputs.
Zeng et al. (2023) showed that simple linear models can rival Transformers on long-horizon benchmarks, a reminder to
compare against strong baselines.

**Transfer learning.** Pretraining on related series and fine-tuning is useful when the target series is short
(Dong et al., 2025, Sec. 8); we test this with ten other PJM regions as the source domain.

**Survey.** Dong et al. (2025) review deep-learning STLF across data, preprocessing, models and evaluation; we follow
their recommendations on z-score scaling, 3σ outlier inspection, MAE/RMSE/MAPE metrics and chronological evaluation.

## 6. Dataset

We use the **PJM East hourly load** series (`PJME_hourly.csv`) from the public *Hourly Energy Consumption* release
(Kaggle `robikscube/hourly-energy-consumption`; IEEE DataPort DOI 10.21227/wqg6-9917). `data/download.py` fetches a
pinned mirror and verifies the SHA-256 fingerprint, so every team member uses the identical file. For the
pretraining study we also download ten other PJM regions from the same release. Full details are in
`docs/dataset_documentation.md`.

**Table 1 – Dataset comparison**

| Dataset | Source | Size | Format | Classes | Licence | Role |
|---|---|---|---|---|---|---|
| **PJM East (PJME)** | PJM via Kaggle *Hourly Energy Consumption* | 145,366 hourly rows, 2002-01-01 → 2018-08-03 | CSV (`Datetime`, `PJME_MW`) | n/a (regression) | CC0: Public Domain (Kaggle); data owner PJM | Train / validation / test |
| 10 other PJM regions (AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJMW) | same release | 33k–121k hourly rows each; only hours ≤ 2014-12-31 used | CSV | n/a | CC0: Public Domain (same release) | External: pretraining only |
| ISO-NE (considered) | ISO New England | Hourly load + temperature 2003–2014 | Excel per year | n/a | ISO-NE terms | Not used |
| GEFCom2014 (considered) | Competition data | Hourly load + 25 weather stations | CSV | n/a | Competition terms | Not used |

The series ranges from 14,544 MW (Hurricane Sandy, 30 Oct 2012) to 62,009 MW (heatwave, 2 Aug 2006), mean 32,080 MW,
with clear daily, weekly and yearly cycles (Fig. 1).

{{fig:experiments/eda/fig1_series_split.png|Fig. 1 – PJM East hourly load and the chronological train / validation / test split.}}

## 7. Data Preprocessing

All steps are in `data/preprocess.py`; statistics are written to `data/processed/data_report.json`.

1. **Sorting** – the raw file is not in time order.
2. **Clock changes** – 4 duplicated timestamps (DST end) are averaged; 30 missing hours (DST start) are filled by
   linear interpolation (≤ 3 h) and flagged in an `imputed` column.
3. **Outliers** – the 3σ rule flags 1,263 hours, but **none are removed**: they are real heatwave/cold-snap peaks and
   Hurricane Sandy, exactly the events a forecaster has to handle. Removing them would make the test set easier.
4. **Calendar features** – cyclical sin/cos encodings put 23:00 next to 00:00 and December next to January;
   `is_weekend`; `is_holiday` from the `holidays` package (US federal, observed dates). A 45-dimensional one-hot
   alternative is kept for the representation study.
5. **Chronological split** – train 2002-01-01 → 2014-12-31 (78.4 %), validation 2015–2016 (12.1 %), test
   2017-01-01 → 2018-08-03 (9.6 %). Never shuffled.
6. **Scaling** – z-score with mean 32,287.5 MW and SD 6,452.9 MW computed **on training rows only**.
7. **Windowing** (`src/datasets/windows.py`) – a sample is 168 input hours → 24 target hours. A sample belongs to a
   split only if **all 24 target hours** are inside that split; inputs may reach back into an earlier split (genuine
   past) but never into a later one. Training uses every hour as an origin (113,760 windows); validation and test use one
   forecast per day issued at 00:00 (731 and 579 forecasts).

`experiments/check_split.py` asserts all of these rules (targets inside their split, no input from a later split,
strict ordering of split boundaries, scaler = training statistics) and prints *ALL LEAKAGE CHECKS PASSED*.

{{fig:experiments/eda/fig2_profile_daytype.png|Fig. 2 – Mean daily profile by day type (training data only). Weekday peaks are ≈ 4 GW above weekend peaks; holidays sit in between.}}

{{fig:experiments/eda/fig3_autocorrelation.png|Fig. 3 – Autocorrelation of the training series. Peaks every 24 h and a local maximum at 168 h justify the one-week lookback.}}

## 8. Methodology

All models share **one data pipeline, one training script (`src/train.py`) and one evaluation script
(`src/evaluate.py`)**, so differences in results come from the models and not from the protocol (Core Req. 6).

* **Strategy.** Direct multi-output (MIMO): one forward pass returns the 24 loads. Deep models use an
  encoder–decoder layout where the decoder receives the 24 **known future calendar vectors** plus a learned
  horizon-step embedding and never consumes its own predictions.
* **Normalisation.** Inputs are globally z-scored; in addition, every deep model is wrapped in **RevIN** (Kim et al.,
  2022, hand-written in `src/models/revin.py`): the load channel of each input window is normalised by the window's own
  mean and SD, and the output is de-normalised with the same statistics. The model then learns the *shape* of the next
  day relative to last week, which makes it robust to the lower average load of 2017–18 compared with 2002–14.
* **Training protocol (identical for all deep models).** AdamW (Loshchilov & Hutter, 2019), L1 loss (matches the
  primary metric), batch 256, 8,192 randomly drawn training windows per epoch, at most 60 epochs, early stopping with
  patience 8 on validation MAE, ReduceLROnPlateau (factor 0.5, patience 3), gradient clipping at 1.0, weight decay 1e-4.
  The checkpoint with the best validation MAE is restored. Training is resumable from a per-epoch state file.
* **Tuning budget.** Every deep model receives the same learning-rate grid {3e-4, 1e-3, 3e-3} evaluated on validation
  only (Section 12). The test set is used once, for the final configurations.
* **Seeds.** Each final configuration is trained with seeds 0, 1, 2; we report mean ± SD.
* **Statistical testing.** Hypotheses are tested with a paired bootstrap over the 579 test days (2,000 resamples) of the
  difference in daily MAE between two models (seed-averaged). A hypothesis is *supported* if the 95 % CI excludes zero in
  the predicted direction, *rejected* if it excludes zero in the opposite direction, and *inconclusive* otherwise.

## 9. Baseline

* **Seasonal naive (weekly):** ŷ<sub>t+h</sub> = y<sub>t+h−168</sub> ("same hour last week").
* **Seasonal naive (daily):** ŷ<sub>t+h</sub> = y<sub>t+h−24</sub> ("same hour yesterday"), shown for context.
* **LightGBM** (Ke et al., 2017), the main baseline: a single "pooled direct" gradient-boosted tree model for all 24
  horizons. Each row is one (origin, horizon *h*) pair with features *h*, lag-168 and lag-24 of the target hour, the last
  observed load, 24 h mean / max / min / SD, 168 h mean, the 24 h trend, and the 10 calendar features of the target hour.
  L1 objective, learning rate 0.05, 63 leaves, up to 2,000 rounds with early stopping (100) on the validation set,
  training origins sub-sampled with stride 4.

## 10. Proposed / Advanced Models

**Classic – LSTM encoder–decoder** (`src/models/lstm.py`). An LSTM encoder (hidden 64) reads the 168 past hours
(load + calendar); its final (h, c) state initialises an LSTM decoder that reads the 24 future calendar vectors plus a
4-dimensional horizon embedding; a linear head outputs one load per hour.

**Modern – patch Transformer with hand-written attention** (`src/models/transformer.py`, `attention.py`). The 168
past hours are cut into **7 daily patches** of 24 h (PatchTST-style), linearly embedded to d = 64 and given sinusoidal
positional encodings; 2 pre-norm encoder blocks apply multi-head self-attention (4 heads) across days. The decoder
has 24 query tokens (projection of each target hour's calendar + learned horizon embedding); 2 decoder blocks apply
self-attention among the 24 hours and **cross-attention into the 7 encoded days**, so the model can learn *which* past
day matters (e.g. the same weekday last week). Scaled dot-product attention, the head split/merge, positional
encoding, patch embedding and the blocks are all written from the equations — no `nn.MultiheadAttention` or
`nn.Transformer` is used (Core Req. 5).

**Advanced – Temporal Fusion Transformer** (`src/models/tft.py`), a compact re-implementation of Lim et al. (2021):
gated linear units and gated residual networks; variable-selection networks that learn a softmax weight per input
variable at every step for both past and known-future inputs; an LSTM encoder (past) and decoder (future) for local
processing; interpretable multi-head attention (heads share the value projection and are averaged) with a causal
mask; gated skip connections and a linear output layer. Static covariates are omitted (single series).

**Transfer learning – pretrained Transformer.** The same Transformer is first trained for 20 epochs on windows from
**ten other PJM regions** (each z-scored with its own statistics, hours ≤ 2014-12-31 only, so the corpus never overlaps
the validation/test period in time), then fine-tuned on PJM East with learning rate 3e-4.

**Table 2 – Model architecture comparison**

| Model | Family | Input | Main layers | Parameters | Pretrained |
|---|---|---|---|---|---|
| Seasonal naive (week / day) | Rule | Lag 168 / 24 | none | 0 | – |
| LightGBM | Gradient-boosted trees | 20 tabular features per (origin, h) | 288 trees, 63 leaves | 288 trees | No |
| LSTM enc-dec | Recurrent (classic) | 168×11 past, 24×10 future | LSTM(64) encoder, LSTM(64) decoder, linear head, RevIN | 40,355 | No |
| Patch Transformer | Attention (modern) | 168×11 past → 7 patches, 24×10 future | Patch embed, 2 enc + 2 dec blocks, d=64, 4 heads, d<sub>ff</sub>=128, RevIN | 186,947 | No |
| TFT | Gated attention hybrid (advanced) | 168×11 past, 24×10 future | VSNs, GRNs, LSTM enc/dec (32), interpretable 4-head attention, RevIN | 44,401 | No |
| Transformer fine-tuned | Attention + transfer | as Transformer | as Transformer | 186,947 | Yes – 10 PJM regions |

## 11. Experimental Setup

* **Hardware.** All final models (Tables 4, 5, 7) were trained on CPU only: an Apple-silicon Mac running a 4-core arm64
  Linux VM, PyTorch 2.5.1, one thread per run and four runs in parallel. The hyperparameter and data-strategy screening
  runs were completed on an Apple M2 (8 cores, macOS, 1 thread per run, 6 runs in parallel). No GPU was used.
* **Software.** Python 3.10–3.12, PyTorch 2.5.1, LightGBM 4, pandas, NumPy, holidays, matplotlib, Gradio.
* **Experiment registry.** Every run (49 in total) is declared in `experiments/plan.yaml` and executed by
  `experiments/run_plan.py`, which runs unfinished runs in parallel and resumes interrupted ones. Each run writes its
  config, training history, best checkpoint, validation/test predictions, metrics and cost to
  `results/runs/<name>/seed<k>/`.
* **Test protocol.** 579 day-ahead forecasts (2017-01-01 → 2018-08-02), one per day at 00:00. Tuning runs are flagged
  `test: false` and never touch the test set.

## 12. Hyperparameters

**Table 3 – Hyperparameter configuration (final models)**

| | LightGBM | LSTM | Transformer | TFT | Transformer fine-tuned |
|---|---|---|---|---|---|
| Optimiser | GBDT | AdamW | AdamW | AdamW | AdamW |
| Learning rate | 0.05 | 1e-3 | 1e-3 | 1e-3 | 3e-4 (pretrain 1e-3) |
| Batch size | – | 256 | 256 | 256 | 256 |
| Epochs / rounds | ≤ 2,000 rounds, early stop 100 | ≤ 60, patience 8 | ≤ 60, patience 8 | ≤ 60, patience 8 | ≤ 60 (pretrain 20) |
| Samples per epoch | stride-4 origins | 8,192 windows | 8,192 windows | 8,192 windows | 8,192 windows |
| Regularisation | bagging 0.8, feature fraction 0.9, min 50 per leaf | dropout 0.1, wd 1e-4, clip 1.0 | dropout 0.1, wd 1e-4, clip 1.0 | dropout 0.1, wd 1e-4, clip 1.0 | as Transformer |
| Loss | L1 | L1 | L1 | L1 | L1 |
| LR schedule | – | ReduceLROnPlateau ×0.5 / 3 ep | same | same | same |
| Augmentation | none | none | none | none | cross-region pretraining |

**Hyperparameter study (Core Req. 7a).** With seed 0 and validation data only, every deep model was trained with the
same learning-rate grid {3e-4, 1e-3, 3e-3}; for the Transformer we additionally varied model width (d = 32 / 64 / 128)
and dropout (0 / 0.1 / 0.3). To give every setting the same budget, the table compares the best validation MAE reached
within the first 30 epochs.

{{table:table_hp_study}}

**Findings.** (i) **Learning rate:** 3e-4 is clearly too slow for every model within the budget (+90 to +190 MW).
For the LSTM and TFT, 3e-3 reached a lower validation MAE than our default 1e-3 (LSTM 1,275 vs 1,351 MW; TFT 1,249 vs
1,313 MW); for the Transformer, 1e-3 and 3e-3 are tied (1,265 vs 1,269 MW). (ii) **Width:** halving the Transformer
(d = 32, 53k parameters) costs 103 MW; doubling it (d = 128, 702k parameters) gains 52 MW at 3.8× the parameters.
(iii) **Dropout:** 0.3 hurts badly (+247 MW) and dropout 0 is the best single setting (1,176 MW): with 113k training
windows and L1 loss the model is not in an over-fitting regime.

**Honest caveat.** The final configurations (lr 1e-3, d = 64, dropout 0.1 for every deep model) were fixed *before*
this screening finished, and the screening uses one seed and a 30-epoch budget that favours faster learning rates. The
study therefore shows that the LSTM and TFT results in Table 4 are probably slightly pessimistic (a higher learning rate
helps them most), and that a smaller dropout would help the Transformer. Every model received the *same* grid, so the
comparison stays fair in budget, but not every model was retrained at its best screened setting (see Limitations).

## 13. Results

**Table 4 – Main results on the test set** (579 day-ahead forecasts; mean ± SD over seeds; 95 % CI from a bootstrap
over test days of the seed-averaged daily MAE).

{{table:table4_main_results}}

The Transformer has the lowest test MAE (1,389 ± 12 MW, MAPE 4.28 %), followed closely by its pretrained-and-fine-tuned
variant (1,394 ± 17 MW). TFT (1,432 MW) and LightGBM (1,440 MW) are statistically tied, and the LSTM (1,460 MW) is
slightly worse than LightGBM on MAE but better on RMSE (2,067 vs 2,109 MW), i.e. it makes fewer very large errors.
All learned models more than halve the error of the weekly naive rule (3,511 MW). Validation and test rankings agree,
which suggests model selection on validation did not overfit. Seed variance is small for the Transformer (SD 12 MW)
and largest for the LSTM (43 MW).

{{fig:experiments/figures/model_comparison.png|Fig. 4 – Test MAE of every model (mean ± SD over seeds).}}

**Per-class results.** Table 4b splits the test error by day type and by season (Core Req. 9).

{{table:table4b_per_class}}

Summer is the hardest season for every model (≈ 1,780–1,860 MW for the deep models): air-conditioning load depends
on temperature, which no model observes. Autumn and spring are easiest. On holidays (n = 17) LightGBM is best
(1,286 MW), the Transformer variants are close (1,347–1,367 MW), and the LSTM is worst (1,604 MW).

**Per-horizon results.**

{{table:table4c_per_horizon}}

{{fig:experiments/figures/per_horizon_mae.png|Fig. 5 – Test MAE across the 24-hour horizon. Error grows from the first hours (close to the last observation) to the evening peak (h ≈ 18) and falls again at night.}}

Error is not monotonic in *h*: it rises from ≈ 420–570 MW at 01:00 to ≈ 2,150 MW at the evening peak (18:00) and
drops again at night, because the peak hours carry the most weather-driven variance. The attention models gain most
in the first half of the day (h1–12: Transformer 827 MW vs LSTM 959 MW and LightGBM 911 MW); in the second half all
learned models converge to ≈ 1,950–1,970 MW.

{{fig:experiments/figures/learning_curves.png|Fig. 6 – Validation MAE during training (seed 0). The Transformer converges fastest; all deep models pass the LightGBM validation level.}}

## 14. Model Comparison

**Hypothesis verdicts** (paired bootstrap over test days; negative Δ = first model better):

{{table:table_hypotheses}}

* **H1 – supported.** The Transformer beats LightGBM by 51 MW (95 % CI [−98, −4]) and the gap exceeds 2 × seed SD
  (24 MW). The margin is modest (3.5 %): a well-featured gradient-boosting model is a strong baseline for day-ahead load,
  consistent with Zeng et al. (2023). The LSTM and TFT do *not* beat LightGBM on MAE.
* **H3 – inconclusive (and the opposite pattern appears).** For h13–24 the Transformer is only 11 MW better than the LSTM
  (CI [−47, +27]). Its advantage is in the *early* hours (−132 MW for h1–12). Attention over daily patches helps the
  model align the start of the day with the matching hours of previous days; for the evening hours, both models are
  limited by the same missing information (weather).
* **H2 and H4** are discussed with the ablation (Section 15) and the pretraining study below.

**Why does the Transformer win?** (1) Patching turns 168 noisy hourly steps into 7 day-tokens, so attention operates
directly on "which previous day looks like tomorrow"; (2) cross-attention from each target hour into those days lets
the model copy the right daily shape (e.g. last Monday for a Monday); (3) with RevIN the network only has to predict
the shape relative to the window, which transfers across years with different average load. The LSTM must compress
168 steps into one state vector, and TFT, although expressive, is harder to optimise in the fixed 60-epoch budget
(its learning curve is still falling at epoch 60).

**Data-strategy study (Core Req. 7b, representation).**

{{table:table_data_strategy}}

* **Normalisation:** removing RevIN is the largest degradation in the study (val +126 MW, test +131 MW; see Section 15).
* **Calendar representation:** the **one-hot calendar (45 features) is better than the cyclical sin/cos encoding** on
  validation (1,182 ± 4 vs 1,205 ± 18 MW) and on test (**1,340 ± 10 MW, MAPE 4.13 %** vs 1,389 ± 12 MW). The paired
  bootstrap difference is −48 MW (95 % CI [−65, −33]), and the one-hot Transformer beats LightGBM by 99 MW
  (CI [−141, −56]). We expected sin/cos to win because it places neighbouring hours close together, but with only
  24 hours and 7 weekdays the network can learn any per-hour and per-weekday shape directly from one-hot indicators, while
  a single sin/cos pair forces a smooth, sinusoidal relationship. Because this variant was selected on **validation**
  data, it is a legitimate final choice: **the one-hot Transformer is the best configuration found in this project.**
  Table 4 keeps the cyclical model as the pre-registered "final" configuration so that all ablations refer to the same
  model.
* **Lookback (1 seed, validation only):** 72 h (1,209 MW) and 336 h (1,225 MW) are both no better than 168 h
  (1,205 ± 18 MW over 3 seeds). One week contains the daily and weekly cycle; a second week adds noise from older
  weather, three days loses the same-weekday-last-week reference without a measurable penalty here.

**Pretrained vs scratch (Core Req. 7c).**

{{table:table_pretrained_vs_scratch}}

With all 13 training years, pretraining on the ten other regions gives **no measurable gain** (+5 MW, CI [−9, +20]):
13 years of target data already contain every pattern the source regions could teach. In the **low-data regime**
(only 2013–2014 PJM East data) pretraining reduces test MAE from 1,608 to 1,415 MW (**−12 %**, CI [−223, −163]) and nearly
matches the model trained on 13 years. The zero-shot pretrained model (never trained on PJM East) already reaches
1,546 MW — better than the seasonal naive rules — showing that load dynamics transfer well between neighbouring
regions once each window is normalised. **H4 is therefore supported for the low-data case and inconclusive for the
full-data case.**

## 15. Ablation Study

**Table 5 – Ablation study** (test set, mean ± SD over seeds; Δ relative to the full model).

{{table:table5_ablation}}

* **RevIN is the most important component**: removing it raises MAE by 131 MW (+9.5 %). Without per-window normalisation
  the network sees inputs whose level drifts between the training years and the test years (average PJM East load in
  2017–18 is lower than in 2002–14), so it extrapolates poorly. This is the classic distribution-shift problem RevIN was
  designed for (Kim et al., 2022).
* **Calendar features (H2)**: removing all 10 calendar features raises MAE by 31 MW overall, with the increase four times
  larger on weekends and holidays (+64 MW) than on weekdays (+16 MW) — exactly the pattern predicted by H2. However, the
  all-days CI [−5, +67] MW includes zero, so with 579 test days the effect is **not statistically significant: H2 is
  inconclusive, with the direction as predicted**. The model can recover much of the calendar from the load history itself
  (the weekly cycle is visible in the 168-hour input), which is why the effect is small.
* **Decoder cross-attention**: removing it changes MAE by −9 MW, within seed noise. Without cross-attention the
  decoder still receives the encoded days through the residual stream of self-attention over target-hour tokens
  combined with the calendar; for this short horizon and single series, the explicit day-to-hour attention is not
  needed. We keep it because it makes the model's use of the past interpretable.
* **TFT interpretable attention** (1 seed): removing the attention layer gives 1,417 MW vs 1,432 ± 28 MW for the full TFT
  (−15 MW, within the seed SD). Like the Transformer's cross-attention, the attention layer of TFT is not what makes it
  accurate on this task: the LSTM encoder–decoder and the gated variable selection already capture the daily and weekly
  structure. Its value is interpretability (which past hours the model attends to), not accuracy.

**Interpretation.** The components that matter are the ones that encode a **prior about the data** — per-window
normalisation (+9.5 %) and calendar information (+2.2 %, mostly on weekends and holidays). Attention layers, the
most "modern" part of both attention models, can be removed without a measurable loss on a 24-hour horizon with a
one-week input.

## 16. Error Analysis

We analyse the seed-averaged Transformer forecast on each of the 579 test days (`experiments/error_analysis.py`).
Days are assigned to the first matching category; a "large-error day" has daily MAE > mean + 1 SD.

**Table 6 – Error analysis**

{{table:table6_error_analysis}}

**The 10 worst test days**

{{table:worst_days}}

{{fig:experiments/figures/worst_days.png|Fig. 7 – The four worst test days: actual load (black), Transformer (blue) and LightGBM (grey, dashed).}}

{{fig:experiments/figures/error_by_month_and_peak.png|Fig. 8 – Left: mean daily MAE by month (summer is hardest). Right: forecast peak minus actual peak against the actual peak.}}

**Findings.**

1. **Weather changes dominate the worst days.** 8 of the 10 worst days are days on which the daily mean load changed
   by more than 2.5 GW from the previous day. Most are **over-forecasts after a hot spell breaks** (e.g. Fri 6 Jul 2018:
   +5.2 GW bias, peak over-forecast by 11 GW; Thu 28 Sep 2017; Wed 23 Aug 2017): the model sees a week of high
   air-conditioning load and expects it to continue, but a cold front arrives overnight. Load history at 00:00 contains
   no signal about tomorrow's temperature, so no load-only model can predict these days — LightGBM fails on the same
   days (MAE 4.3–4.7 GW).
2. **Extreme peaks are under-forecast.** On the top-10 % peak-load days the mean bias is −572 MW, and across all days the
   peak error correlates negatively with the actual peak (r = −0.25; mean peak error on the hottest days −1.7 GW). The first
   day of a heatwave (Mon 19 Jun 2017, Wed 14 Jun 2017: actual peaks 48.2 and 46.6 GW) is under-predicted because the
   preceding week was cooler. Interestingly LightGBM is much better on these two days (2.6 GW), because its explicit
   *max24* and *trend24* features react faster to a rising level than the RevIN-normalised Transformer.
3. **Holidays and bridge days.** Holidays have only a slightly higher error than normal days (1,352 vs 1,126 MW), but with a
   negative bias (−364 MW): the single `is_holiday` bit cannot tell Veterans Day (most businesses open — worst holiday,
   2,797 MW) from Christmas (almost everything closed). Bridge days (the day before/after a holiday) are not encoded at all.
4. **Normal days** (351 of 579) have a mean error of 1,126 MW, about 3.5 % of the load — the irreducible noise of a
   load-only model.
5. **Daylight-saving days** are handled well (833 MW) thanks to the regular hourly grid built in preprocessing.

**Implications.** Most of the remaining error is caused by **information the model does not have** (tomorrow's
temperature), not by insufficient capacity. The most promising improvement is therefore weather *forecasts* as known
future inputs (Section 21), followed by richer holiday encoding (holiday type, bridge days).

## 17. Discussion

* **Deep learning helps, but modestly, over a strong baseline.** The best deep model is 3.5 % better than LightGBM and
  60 % better than the weekly naive rule. The gain is significant but small, in line with recent evidence that
  well-engineered tree models and simple models are hard to beat on single-series forecasting (Zeng et al., 2023).
* **Inductive bias matters more than attention.** Width does help (d = 32 → 64 → 128 lowers validation MAE from 1,368 to
  1,265 to 1,213 MW), but the largest effects come from components that encode a prior about the data: per-window
  normalisation (−9.5 %), the calendar representation (one-hot −3.5 %) and calendar inputs on weekends and holidays.
  Removing attention (in either attention model) changes nothing measurable.
* **Normalisation is the unsung hero.** The largest single effect in the whole study is RevIN (−9.5 % error), larger than
  the difference between any two deep architectures. Distribution shift over the 16 years of data is the real enemy.
* **Transfer learning is a data-efficiency tool.** Pretraining does not improve a model that already has 13 years of data,
  but it lets a model trained on 2 years almost match one trained on 13.
* **Cost.** LightGBM trains in ≈ 1.5 min and infers in 0.2 ms per forecast; the Transformer needs ≈ 9 min on one CPU thread and
  0.37 ms per forecast (batched) — both are negligible for a once-a-day forecast. TFT is the most expensive (≈ 30 min,
  1.5 ms) without being more accurate.

**Table 7 – Computational cost** (final models, CPU only).

{{table:table7_cost}}

## 18. Limitations

* **No weather inputs.** The models see only load and calendar. Operational forecasters use temperature forecasts; our
  error analysis shows this is the main source of remaining error. Using *measured* future temperature would have been
  leakage, and historical weather forecasts are not available for this dataset.
* **Single region and period.** Results are for PJM East 2017–2018; the ranking of models may differ for other grids,
  for residential loads, or for longer horizons.
* **Point forecasts only.** We do not produce prediction intervals, which operators need for reserve planning.
* **Limited tuning budget, and final configurations fixed before the screening.** Each deep model received the same
  small grid and a 60-epoch budget, but the final models were trained with the default settings (lr 1e-3, dropout 0.1),
  while the screening later showed lr 3e-3 to be better for the LSTM and TFT and dropout 0 / one-hot calendar to be better
  for the Transformer. Retraining every model at its best screened setting with 3 seeds is the obvious next step; the
  LSTM and TFT numbers in Table 4 are probably slightly pessimistic.
* **Ablations of TFT and the hyperparameter/representation screens use one seed** to save CPU time; differences smaller
  than the seed SD of the main models (≈ 12–43 MW) should not be over-interpreted.
* **Mixed hardware for timings.** The final models were timed on a 4-core Linux VM and the screening runs on an M2 Mac; we
  only compare training times within the same hardware (Table 7).

## 19. Ethical Considerations

* **Privacy.** The data is aggregated regional load (tens of GW); it contains no household or personal information.
* **Use of forecasts.** Forecasts feed decisions with economic and safety consequences. Systematic under-forecasting of
  heatwave peaks (Section 16) is the most safety-relevant failure: it could lead to insufficient reserves exactly when the
  grid is most stressed. A deployed system should report uncertainty and keep a human operator in the loop.
* **Environmental cost.** All training ran on a laptop CPU; the complete study (49 runs) used roughly 10 CPU-hours.
* **Licences and attribution.** The dataset is a public release of PJM data; every external idea is cited, all model code
  was written by the team, and no external AI API is used for any prediction (Core Req. 13).

## 20. Conclusion

We built a leakage-free, fully reproducible day-ahead load forecasting system for PJM East and compared five model
families under one protocol. A patch Transformer with hand-written attention and RevIN is the most accurate model
(test MAE 1,389 ± 12 MW, MAPE 4.28 %), significantly but modestly better than a strong LightGBM baseline (H1 supported);
with a one-hot calendar, selected on validation in the data-strategy study, it reaches **1,340 ± 10 MW (MAPE 4.13 %)**,
99 MW better than LightGBM.
Calendar features help most on weekends and holidays, but the effect is small (H2 inconclusive); the Transformer's
advantage over the LSTM lies in the early hours, not the late ones (H3 inconclusive); and cross-region pretraining
helps strongly only when target data is scarce (H4 supported in the low-data regime). Per-window normalisation is the
most important single component, and the remaining errors are concentrated on weather-driven days that load history
cannot anticipate.

## 21. Future Work

1. Add **day-ahead temperature forecasts** for the PJM East load-weighted cities as known future inputs.
2. **Probabilistic forecasting** with quantile loss (TFT supports this natively) to give operators prediction intervals.
3. **Richer calendar**: holiday type embeddings, bridge days, school holidays.
4. **Ensembles** of the Transformer and LightGBM, whose errors differ on heatwave onset days.
5. Longer training and a larger search for TFT; multi-region training with region embeddings.

## 22. References

1. Dong, Q., Huang, R., Cui, C., Towey, D., Zhou, L., Tian, J., & Wang, J. (2025). Short-term electricity-load forecasting
   by deep learning: A comprehensive survey. *Engineering Applications of Artificial Intelligence*.
   https://doi.org/10.1016/j.engappai.2025.110980
2. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
4. Hong, T., & Fan, S. (2016). Probabilistic electric load forecasting: A tutorial review. *International Journal of
   Forecasting*, 32(3), 914–938.
5. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
6. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient
   gradient boosting decision tree. *NeurIPS 30*.
7. Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible instance normalization for accurate
   time-series forecasting against distribution shift. *ICLR*.
8. Kong, W., Dong, Z. Y., Jia, Y., Hill, D. J., Xu, Y., & Zhang, Y. (2019). Short-term residential load forecasting based
   on LSTM recurrent neural network. *IEEE Transactions on Smart Grid*, 10(1), 841–851.
9. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time
   series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.
10. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
11. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting
    with Transformers. *ICLR*.
12. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *NeurIPS 27*.
13. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
    Attention is all you need. *NeurIPS 30*.
14. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers effective for time series forecasting? *AAAI*.
15. PJM Interconnection LLC. Hourly load data, public release *Hourly Energy Consumption* (Kaggle,
    `robikscube/hourly-energy-consumption`); IEEE DataPort DOI 10.21227/wqg6-9917.

## Appendix A – Reproducing every number in this report

```bash
python data/download.py && python data/preprocess.py && python experiments/check_split.py
python experiments/run_plan.py --workers 4 --threads 1     # all 49 runs (resumable)
python src/evaluate.py && python experiments/error_analysis.py
python report/build_report.py forecast_AI
```
