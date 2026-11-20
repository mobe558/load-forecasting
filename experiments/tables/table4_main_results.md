### Table 4 - Main results on the test set (mean ± SD over seeds)

| Model                                   |   Seeds | Test MAE (MW)   | 95% CI (MW)    | RMSE (MW)   | MAPE (%)     |   Val MAE (MW) |
|:----------------------------------------|--------:|:----------------|:---------------|:------------|:-------------|---------------:|
| Seasonal naive (last week)              |       1 | 3,511 ± 0       | [3,289, 3,735] | 4,794 ± 0   | 10.99 ± 0.00 |          3,312 |
| Seasonal naive (yesterday)              |       1 | 2,300 ± 0       | [2,179, 2,422] | 3,134 ± 0   | 7.32 ± 0.00  |          2,137 |
| LightGBM (baseline)                     |       3 | 1,440 ± 3       | [1,362, 1,519] | 2,109 ± 3   | 4.45 ± 0.01  |          1,279 |
| LSTM enc-dec (classic)                  |       3 | 1,460 ± 43      | [1,392, 1,531] | 2,067 ± 50  | 4.54 ± 0.14  |          1,285 |
| Transformer (modern, hand-written attn) |       3 | 1,389 ± 12      | [1,316, 1,467] | 2,036 ± 15  | 4.28 ± 0.04  |          1,205 |
| TFT (advanced)                          |       3 | 1,432 ± 28      | [1,362, 1,508] | 2,053 ± 32  | 4.43 ± 0.08  |          1,243 |
| Transformer pretrained+fine-tuned       |       3 | 1,394 ± 17      | [1,322, 1,471] | 2,037 ± 11  | 4.31 ± 0.06  |          1,215 |
