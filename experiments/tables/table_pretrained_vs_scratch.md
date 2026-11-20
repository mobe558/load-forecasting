### Pretrained vs scratch (Transformer)

| Init                               | PJME training data   |   Seeds | Test MAE (MW)   | MAPE (%)    |   Train time (s) |
|:-----------------------------------|:---------------------|--------:|:----------------|:------------|-----------------:|
| scratch                            | full (2002-2014)     |       3 | 1,389 ± 12      | 4.28 ± 0.04 |              536 |
| pretrained + fine-tuned            | full (2002-2014)     |       3 | 1,394 ± 17      | 4.31 ± 0.06 |              534 |
| pretrained only, zero-shot on PJME | none (other regions) |       1 | 1,546 ± 0       | 4.80 ± 0.00 |              184 |
| scratch                            | 2 years (2013-2014)  |       3 | 1,608 ± 19      | 5.01 ± 0.09 |              264 |
| pretrained + fine-tuned            | 2 years (2013-2014)  |       3 | 1,415 ± 9       | 4.37 ± 0.03 |              295 |
