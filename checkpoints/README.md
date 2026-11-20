# Trained checkpoints

The final checkpoints are small (≈ 43 MB in total), so they are stored **in the repository** next to the run that
produced them, instead of in this folder:

| Model | Checkpoint(s) |
|---|---|
| Transformer (best model) | `results/runs/transformer/seed{0,1,2}/model.pt` |
| Transformer pretrained on 10 regions | `results/runs/transformer_pretrain/seed0/model.pt` |
| Transformer fine-tuned | `results/runs/transformer_finetune/seed{0,1,2}/model.pt` |
| LSTM enc-dec | `results/runs/lstm/seed{0,1,2}/model.pt` |
| TFT | `results/runs/tft/seed{0,1,2}/model.pt` |
| LightGBM | `results/runs/lightgbm/seed{0,1,2}/model.txt` |

Each `model.pt` is a PyTorch `state_dict`; the matching architecture and hyperparameters are in `config.json` in the
same folder. Load one with `src/predict.py` (`DLForecaster(run_dir)`), or simply run `python demo/app.py`.

To reproduce every checkpoint from scratch:

```bash
python experiments/run_plan.py --workers 4 --threads 1
```

Shared-drive link (optional mirror of `results/runs/`): _add link here_
