# Core Requirement 1 - Problem Definition

**Project idea #13 - Multi-horizon electricity load forecasting (time series + calendar -> forecast)**

## 1. Task

Day-ahead, multi-horizon **point forecasting** of the hourly electricity load of the PJM East region (USA).
Every day at 00:00 the model issues one forecast for **all 24 hours of that day at once**.

This is *Short-Term Electricity-Load Forecasting (STELF)*: using historical load data to predict
the load over the next few hours to a few days, mainly for short-term grid scheduling
(Dong et al., 2025, Sec. 1). "Multi-horizon" means that the output is a vector (one value per future
hour, h = 1 ... 24) and not a single next value, so each forecast must stay accurate from hour 1 to hour 24.

## 2. Input

| Part | Shape | Content | Why it is allowed |
|---|---|---|---|
| Past load | 168 x 1 | Load (MW, z-score scaled) for the previous 7 days, hours t-168 ... t-1 | Already observed when the forecast is made |
| Past calendar | 168 x 10 | Calendar features of those same hours | Known |
| Future calendar | 24 x 10 | Calendar features of the 24 hours being forecast | **Known in advance** (a calendar is fixed), so this is not leakage |

The 10 calendar features (default *cyclical* encoding): `hour_sin, hour_cos, dow_sin, dow_cos,
month_sin, month_cos, doy_sin, doy_cos, is_weekend, is_holiday`.
The alternative *one-hot* encoding (24 + 7 + 12 + 2 = 45 features) is kept for the data-representation study.

**Why 168 hours?** The autocorrelation of the training series peaks at lag 24 (0.89) and has a local peak at
lag 168 (0.79, higher than lags 144 = 0.75 and 192 = 0.74). One week of history therefore contains both the daily
and the weekly cycle (`experiments/eda/fig3_autocorrelation.png`).

## 3. Output

A vector **y_hat = (y_hat(t), ..., y_hat(t+23))** of the 24 hourly loads of the forecast day, in MW
(the model predicts scaled values, which are converted back with the training mean and standard deviation).
The strategy is **direct multi-output (MIMO)**: the 24 values come out of one forward pass. The model does not
feed its own predictions back in hour by hour (recursive), so errors cannot pile up step by step.

## 4. Evaluation metrics

Following the metrics most used in the STELF literature (Dong et al., 2025, Sec. 10):

* **Primary:** MAE in MW, averaged over the 24 horizons and all test days.
* **Secondary:** RMSE (punishes big misses at peaks), MAPE in % (scale-free, comparable to other papers),
  and **MAE per horizon step h = 1 ... 24** (the multi-horizon view).
* Every model is trained with **3 random seeds** and reported as mean +/- standard deviation.

## 5. Hypotheses

| # | Hypothesis (falsifiable) | Experiment that tests it | Confirmed if ... | Rejected if ... |
|---|---|---|---|---|
| **H1** | Deep learning models forecast more accurately than the simple reference models (seasonal-naive "same hour last week" and a LightGBM regressor). | Main comparison: all models on the same test days, 3 seeds. | The best DL model has lower mean test MAE than the best baseline, and the gap is larger than 2 x seed SD. | A baseline is as good as or better than the DL models. |
| **H2** | Calendar information (hour, weekday, season, holiday) reduces forecast error, most of all on holidays and weekends. | Ablation: retrain the best model with `calendar_encoding: none` (load history only). | MAE rises without calendar features, and the rise is larger on holiday/weekend days than on normal weekdays. | MAE does not change significantly. |
| **H3** | An attention-based model (Transformer) keeps its accuracy better over the horizon than a recurrent model (LSTM): its advantage grows for later hours (h = 13 ... 24). | Per-horizon MAE curves of the LSTM and the Transformer. | The Transformer's MAE for h = 13 ... 24 is lower than the LSTM's, with a gap larger than seed SD. | The LSTM is equal or better at the later horizons. |
| H4 *(optional, linked to Req. 7)* | A pretrained time-series model, fine-tuned on PJM, beats the same architecture trained from scratch. | Pretrained-vs-scratch comparison. | Fine-tuned MAE < from-scratch MAE. | No gain from pretraining. |

Each hypothesis maps onto a later core requirement: H1 -> baseline and model comparison (Req. 3, 4, 6),
H2 -> ablation (Req. 8), H3 -> evaluation per horizon and error analysis (Req. 9, 10), H4 -> experiments (Req. 7).

## Reference

Dong, Q., Huang, R., Cui, C., Towey, D., Zhou, L., Tian, J., & Wang, J. (2025).
*Short-Term Electricity-Load Forecasting by Deep Learning: A Comprehensive Survey.*
Engineering Applications of Artificial Intelligence. https://doi.org/10.1016/j.engappai.2025.110980
(open version: arXiv:2408.16202v2, May 2025).
