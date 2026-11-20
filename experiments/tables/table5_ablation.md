### Table 5 - Ablation study (test set, mean ± SD over 3 seeds)

| Model       | Component removed             | Test MAE (MW)   | Δ vs full (MW)   |   Δ weekdays |   Δ weekends+holidays |
|:------------|:------------------------------|:----------------|:-----------------|-------------:|----------------------:|
| transformer | - (full model)                | 1,389 ± 12      |                  |              |                       |
| transformer | calendar features (all 10)    | 1,420 ± 22      | +31 (+2.2%)      |          +16 |                   +64 |
| transformer | decoder cross-attention       | 1,380 ± 19      | -9 (-0.6%)       |           -6 |                   -14 |
| transformer | RevIN instance normalisation  | 1,520 ± 19      | +131 (+9.5%)     |         +145 |                  +100 |
| tft         | - (full model)                | 1,432 ± 28      |                  |              |                       |
| tft         | interpretable attention layer | 1,417 ± 0       | -15 (-1.1%)      |          -13 |                   -19 |
