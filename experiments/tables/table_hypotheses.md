### Hypothesis verdicts (paired bootstrap over test days, negative Δ = first model better)

| Hypothesis                                                    | Comparison                                                | Δ MAE (MW)      | 95% CI               | Verdict                               |
|:--------------------------------------------------------------|:----------------------------------------------------------|:----------------|:---------------------|:--------------------------------------|
| H1 best DL model beats the baseline                           | transformer - lightgbm                                    | -51             | [-98, -4]            | supported                             |
| H2 calendar features reduce error (most on weekends/holidays) | no-calendar - full (all days / weekend+holiday / weekday) | +31 / +64 / +16 | [-5, +67] (all days) | inconclusive (direction as predicted) |
| H3 Transformer beats LSTM at later horizons (h13-24)          | transformer - lstm (h13-24 / h1-12)                       | -11 / -132      | [-47, +27] (h13-24)  | inconclusive                          |
| H4 pretraining helps (full data)                              | transformer_finetune - transformer                        | +5              | [-9, +20]            | inconclusive                          |
| H4 pretraining helps (2-year data)                            | pt_finetune_2y - pt_scratch_2y                            | -193            | [-223, -163]         | supported                             |
