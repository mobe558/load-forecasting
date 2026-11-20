# Multi-horizon Electricity Load Forecasting (Project idea #13)

**Team:** forecast_AI

| Member | Student ID |
|---|---|
| Mohab Mohamed Moustafa Mahmoud Mohamed Aboklela | 202303375 |
| Rosana Zarif Sadek Semean | 202303306 |
| Yousef Mohamed Ragab Abdo AboEita | 202302137 |
| Avronia Magdy Tawfik Boutros | 202303908 |
| Sandra Samer Samir Fargallah | 202303334 |

Deep Learning AI403 · Pharos University in Alexandria · Fall 2026

Every day at 00:00, forecast the **24 hourly loads of the PJM East grid** (USA) for that day from
the **previous 168 hours of load + calendar information** (hour, weekday, season, US holidays).

**Best result:** the patch Transformer with hand-written attention and RevIN reaches a **test MAE of
1,389 ± 12 MW (MAPE 4.28 %)** over 579 day-ahead forecasts (2017-01-01 … 2018-08-02), 51 MW better than the
LightGBM baseline (paired bootstrap 95 % CI [−98, −4] MW) and 60 % better than "same hour last week".
With a one-hot calendar encoding (selected on validation in the data-strategy study) it improves to
**1,340 ± 10 MW (MAPE 4.13 %)**.
All numbers: `experiments/tables/table4_main_results.md`; full write-up: `report/forecast_AI_Report.pdf`.

## Repository layout

```
load-forecasting/
├── README.md
├── requirements.txt
├── configs/            one YAML per model / experiment (data.yaml = shared split & windowing)
├── data/               download.py, preprocess.py (raw + processed data are NOT committed)
├── src/
│   ├── datasets/       windows.py (leakage-free windows), regions.py (pretraining corpus)
│   ├── models/         baselines.py, lstm.py, transformer.py, tft.py, attention.py, revin.py
│   ├── train.py        one training entry point for every model
│   ├── evaluate.py     shared evaluation: all tables + hypothesis verdicts
│   ├── metrics.py      MAE / RMSE / MAPE / bootstrap CI
│   └── predict.py      inference helpers used by the demo
├── experiments/        plan.yaml (every run), run_plan.py, check_split.py, eda.py, error_analysis.py,
│                       tables/, figures/, eda/
├── results/runs/       one folder per run and seed: config, history, metrics, cost, predictions, model.pt
├── checkpoints/        README with the link to the trained checkpoints
├── demo/app.py         Gradio live demo
├── docs/               problem_definition.md (Req. 1), dataset_documentation.md (Req. 2)
├── report/             report.md -> forecast_AI_Report.pdf, slides.md -> forecast_AI_Slides.pdf
└── contributions/      individual weekly logs
```

## 1. Environment setup

Python 3.10+ (tested with 3.10.12 on Linux and 3.12.5 on macOS, CPU only, no GPU needed).

```bash
cd load-forecasting
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

macOS: LightGBM needs OpenMP (`brew install libomp`). If you forget to activate the venv,
`experiments/run_plan.py` switches to `.venv/bin/python` automatically.

## 2. Dataset download and preprocessing

```bash
python data/download.py            # PJME_hourly.csv (SHA-256 checked) + 10 other PJM regions for pretraining
python data/preprocess.py          # cleaning, calendar features, chronological split, train-only scaler
python experiments/check_split.py  # asserts the split is leakage-free (prints ALL LEAKAGE CHECKS PASSED)
python experiments/eda.py          # dataset figures -> experiments/eda/
```

## 3. Training

Each model on its own (one seed):

```bash
python src/train.py --config configs/naive_week.yaml          # seasonal naive baseline
python src/train.py --config configs/lightgbm.yaml --seed 0   # LightGBM baseline
python src/train.py --config configs/lstm.yaml --seed 0       # classic: LSTM encoder-decoder
python src/train.py --config configs/transformer.yaml --seed 0   # modern: patch Transformer
python src/train.py --config configs/tft.yaml --seed 0        # advanced: Temporal Fusion Transformer
python src/train.py --config configs/transformer_pretrain.yaml --seed 0   # pretrain on 10 other regions
python src/train.py --config configs/transformer_finetune.yaml --seed 0   # fine-tune on PJME
```

**Everything in the report** (all 49 runs: 3 seeds per final model, the hyperparameter, data-strategy,
pretrained-vs-scratch and ablation studies) is listed in `experiments/plan.yaml`:

```bash
caffeinate -i python experiments/run_plan.py --workers 4 --threads 1   # macOS: caffeinate keeps the Mac awake
python experiments/run_plan.py --status                                  # what is done
```

`run_plan.py` prints a line when each run starts/finishes and a progress report (latest epoch of every running job)
every 60 s; the full log of each run is in `results/runs/<name>/seed<k>/train.log`. The slowest runs (LSTM/TFT) are
started first. Use `--workers` ≈ number of performance cores (4 on an M1/M2, 6 also works on an 8-core M2).
Runs are resumable: press Ctrl-C at any time and run the same command again.

## 4. Evaluation

```bash
python src/evaluate.py               # all report tables -> experiments/tables/, figures -> experiments/figures/
python experiments/error_analysis.py # Table 6 + worst-day plots
```

## 5. Live demo

```bash
python demo/app.py                   # open http://127.0.0.1:7860
```

Tab 1 replays any test day (2017-01-01 … 2018-08-02) with the real previous week, and shows every
model's forecast against the actual load. Tab 2 takes your own CSV (`timestamp, load_mw`, at least
168 hourly rows) and forecasts the next 24 hours. The demo needs trained checkpoints in `results/runs/`.

## 6. Report and slides

```bash
python report/build_report.py forecast_AI   # -> report/forecast_AI_Report.pdf + report/forecast_AI_Slides.pdf
```

Sources: `report/report.md` and `report/slides.md`; tables and figures are pulled in from `experiments/` so the
PDFs always show the numbers the code produced. Needs Google Chrome/Chromium (pandoc optional).
Contribution-log template: `contributions/TEMPLATE_Log.md`. Checkpoint locations: `checkpoints/README.md`.

## 7. Hardware and expected runtimes

All results were produced on **CPU only**: an Apple-silicon Mac running a 4-core arm64 Linux VM, PyTorch
2.5.1 CPU, 1 thread per run and 4 runs in parallel.

| Step | Time |
|---|---|
| download + preprocess + checks | ~1 min |
| naive / LightGBM baselines | seconds / ~1 min per seed |
| Transformer, 60 epochs × 8,192 windows | ~7 min per seed |
| LSTM / TFT | ~25–35 min per seed |
| full plan (49 runs) | ~4–5 h with 4 parallel workers (≈ 1 h for the 16 study runs on an M2 with 6 workers) |

Per-model training time, inference time, parameters, size and memory are in
`experiments/tables/table7_cost.md`.

## 8. Credits

- **Data:** PJM Interconnection LLC, hourly load. Public release: Kaggle `robikscube/hourly-energy-consumption`
  (mirror: github.com/panambY/Hourly_Energy_Consumption). IEEE DataPort DOI 10.21227/wqg6-9917. Licence: CC0: Public Domain.
- **Survey guiding the design:** Dong et al. (2025), *Short-Term Electricity-Load Forecasting by Deep
  Learning: A Comprehensive Survey*, Engineering Applications of AI, doi:10.1016/j.engappai.2025.110980.
- **Architectures** (re-implemented from the papers, no code copied): Vaswani et al. (2017) Transformer;
  Nie et al. (2023) PatchTST patching; Lim et al. (2021) Temporal Fusion Transformer; Kim et al. (2022)
  RevIN; Hochreiter & Schmidhuber (1997) LSTM; Ke et al. (2017) LightGBM.
- **Libraries:** PyTorch, LightGBM, pandas, NumPy, matplotlib, holidays, Gradio.
- No external AI API is used for any prediction.
