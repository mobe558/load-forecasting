### Per-class test MAE (MW): by day type and by season

| Model                                   |   weekday (n=399) |   weekend (n=163) |   holiday (n=17) |   winter (n=149) |   spring (n=184) |   summer (n=155) |   autumn (n=91) |
|:----------------------------------------|------------------:|------------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------:|
| Seasonal naive (last week)              |             3,519 |             3,427 |            4,147 |            3,871 |            2,470 |            4,653 |           3,082 |
| Seasonal naive (yesterday)              |             2,178 |             2,557 |            2,709 |            2,330 |            1,960 |            2,941 |           1,847 |
| LightGBM (baseline)                     |             1,427 |             1,487 |            1,286 |            1,468 |            1,260 |            1,788 |           1,163 |
| LSTM enc-dec (classic)                  |             1,448 |             1,475 |            1,604 |            1,453 |            1,282 |            1,786 |           1,276 |
| Transformer (modern, hand-written attn) |             1,381 |             1,408 |            1,367 |            1,322 |            1,216 |            1,779 |           1,181 |
| TFT (advanced)                          |             1,429 |             1,441 |            1,421 |            1,327 |            1,275 |            1,860 |           1,195 |
| Transformer pretrained+fine-tuned       |             1,397 |             1,390 |            1,347 |            1,304 |            1,225 |            1,808 |           1,177 |
