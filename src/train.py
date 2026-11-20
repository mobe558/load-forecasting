"""Train any model of the project with the SAME data pipeline and protocol (Core Req. 5-6, 11).

Usage
-----
  python src/train.py --config configs/transformer.yaml --seed 0
  python src/train.py --config configs/lstm.yaml --seed 1 --set train.lr=0.003 --name lstm_lr3e-3 --no-test

Every run writes to results/runs/<name>/seed<k>/:
  config.json, history.csv, model.pt (best epoch), preds_val.npz, preds_test.npz,
  metrics.json (val/test MAE, RMSE, MAPE, per-horizon MAE) and cost.json (Core Req. 11).
Training is resumable: state.pt is saved after every epoch; re-running the same command continues.
`--time-budget S` stops cleanly after the epoch that crosses S seconds (used for short sessions).
"""
import argparse
import copy
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.windows import (CFG as DATA_CFG, LoadWindows, TorchBatcher, load_processed,  # noqa: E402
                                  restrict_origins)
from src.metrics import all_metrics  # noqa: E402

DEFAULT_TRAIN = {"lr": 1e-3, "batch_size": 256, "max_epochs": 60, "patience": 8,
                 "samples_per_epoch": 8192, "weight_decay": 1e-4, "grad_clip": 1.0, "loss": "l1"}


# ------------------------------------------------------------------------------------------- config
def deep_set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def load_config(path, overrides):
    cfg = yaml.safe_load(open(ROOT / path))
    cfg.setdefault("data", {})
    cfg["train"] = {**DEFAULT_TRAIN, **cfg.get("train", {})}
    for o in overrides or []:
        k, v = o.split("=", 1)
        deep_set(cfg, k, v)
    return cfg


def hardware():
    try:
        import torch
        tv, th = torch.__version__, torch.get_num_threads()
    except ImportError:
        tv, th = "n/a", None
    return {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "n/a", "cpu_count": os.cpu_count(),
            "torch_threads": th, "device": "cpu", "python": platform.python_version(), "torch": tv}


# ------------------------------------------------------------------------------------------- data
def make_sources(cfg, df):
    enc = cfg["data"].get("encoding", DATA_CFG["features"]["calendar_encoding"])
    lookback = cfg["data"].get("lookback")
    if cfg["data"].get("source") == "regions":                       # pretraining corpus
        from src.datasets.regions import pretrain_source
        tr = pretrain_source(enc)
    else:
        tr = restrict_origins(LoadWindows(df, "train", encoding=enc, lookback=lookback), df,
                              cfg["data"].get("train_start"))
    va = LoadWindows(df, "val", encoding=enc, lookback=lookback)
    te = LoadWindows(df, "test", encoding=enc, lookback=lookback)
    return tr, va, te


def to_mw(x, scaler):
    return x * scaler["std"] + scaler["mean"]


def save_preds(path, src, pred_scaled, scaler):
    t = src.origins[:, None] + np.arange(src.H)[None, :]
    np.savez_compressed(path, origin=src.timestamps[src.origins].astype("datetime64[h]").astype(str),
                        y_true=to_mw(src.load[t], scaler), y_pred=to_mw(pred_scaled, scaler))


# ------------------------------------------------------------------------------------------- baselines
def run_baseline(cfg, seed, run_dir, srcs, scaler, do_test):
    from src.models.baselines import lgb_rows, seasonal_naive
    from src.datasets.windows import CYCLICAL
    tr, va, te = srcs
    name = cfg["model"]["name"]
    t0 = time.time()
    if name.startswith("naive"):
        season = 168 if name == "naive_week" else 24
        predict = lambda s: seasonal_naive(s, season)  # noqa: E731
        n_params, size = 0, 0
    else:
        import lightgbm as lgb
        p = dict(cfg["model"].get("params", {}))
        stride = p.pop("train_stride", 4)
        rounds = p.pop("num_boost_round", 2000)
        sub = copy.copy(tr)
        sub.origins = tr.origins[::stride]
        Xtr, ytr, names = lgb_rows(sub, CYCLICAL if tr.cal.shape[1] == len(CYCLICAL) else
                                   [f"cal{i}" for i in range(tr.cal.shape[1])])
        Xva, yva, _ = lgb_rows(va, names[10:])
        params = {"objective": "l1", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
                  "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                  "seed": seed, "num_threads": 0, **p}
        booster = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names),
                            rounds, valid_sets=[lgb.Dataset(Xva, yva)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        booster.save_model(str(run_dir / "model.txt"), num_iteration=booster.best_iteration)
        predict = lambda s: booster.predict(lgb_rows(s, names[10:])[0],  # noqa: E731
                                            num_iteration=booster.best_iteration).reshape(len(s), s.H)
        n_params = int(booster.num_trees())
        size = (run_dir / "model.txt").stat().st_size
        imp = dict(zip(names, booster.feature_importance("gain").round(1).tolist()))
        json.dump(imp, open(run_dir / "feature_importance.json", "w"), indent=2)
    train_time = time.time() - t0
    out = {}
    for split, s in [("val", va)] + ([("test", te)] if do_test else []):
        t1 = time.time()
        p_ = predict(s)
        inf = time.time() - t1
        save_preds(run_dir / f"preds_{split}.npz", s, p_, scaler)
        d = np.load(run_dir / f"preds_{split}.npz")
        out[split] = all_metrics(d["y_true"], d["y_pred"])
        out[f"{split}_inference_s"] = inf
    cost = {"params": n_params, "params_note": "trees" if name == "lightgbm" else "none",
            "model_size_mb": size / 1e6, "train_time_s": train_time,
            "inference_ms_per_forecast": 1000 * out.get("test_inference_s", out["val_inference_s"]) /
            len(te if do_test else va), "peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "hardware": hardware()}
    return out, cost


# ------------------------------------------------------------------------------------------- deep models
def predict_dl(model, src, batch=512):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for xp, xf, _ in TorchBatcher(src).batches(batch):
            out.append(model(xp, xf).numpy())
    return np.concatenate(out)


def run_deep(cfg, seed, run_dir, srcs, scaler, do_test, time_budget):
    import torch
    from src.models import build_model
    tr, va, te = srcs
    tc = cfg["train"]
    torch.manual_seed(seed), np.random.seed(seed)
    n_past, n_fut = 1 + tr.cal.shape[1], tr.cal.shape[1]
    model = build_model(cfg["model"]["name"], n_past, n_fut, tr.H, **cfg["model"].get("params", {}))
    if cfg.get("init_from"):                                          # fine-tune a pretrained model
        model.load_state_dict(torch.load(ROOT / cfg["init_from"], map_location="cpu", weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = torch.nn.L1Loss() if tc["loss"] == "l1" else torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    st = {"epoch": 0, "best": float("inf"), "bad": 0, "history": [], "train_time": 0.0}
    state_path, best_path = run_dir / "state.pt", run_dir / "model.pt"
    if state_path.exists():                                           # resume
        s = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]), opt.load_state_dict(s["opt"]), sched.load_state_dict(s["sched"])
        gen.set_state(s["gen"]), st.update(s["st"])
        print(f"resumed at epoch {st['epoch']}")
    batcher, t_start = TorchBatcher(tr), time.time()
    y_va = va.load[va.origins[:, None] + np.arange(va.H)[None, :]]
    last_epoch_s = 0.0
    while st["epoch"] < tc["max_epochs"] and st["bad"] < tc["patience"]:
        if time_budget and time.time() - t_start + 1.3 * last_epoch_s > time_budget:
            print("time budget reached - re-run the same command to resume")
            return None, None
        t0 = time.time()
        model.train()
        losses = []
        for xp, xf, y in batcher.batches(tc["batch_size"], True, tc["samples_per_epoch"], gen):
            opt.zero_grad()
            loss = loss_fn(model(xp, xf), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            losses.append(loss.item())
        val_mae_mw = float(np.abs(predict_dl(model, va) - y_va).mean() * scaler["std"])
        sched.step(val_mae_mw)
        st["epoch"] += 1
        st["train_time"] += time.time() - t0
        st["history"].append({"epoch": st["epoch"], "train_loss": float(np.mean(losses)),
                              "val_mae_mw": val_mae_mw, "lr": opt.param_groups[0]["lr"],
                              "epoch_s": time.time() - t0})
        if val_mae_mw < st["best"]:
            st["best"], st["bad"] = val_mae_mw, 0
            torch.save(model.state_dict(), best_path.with_suffix(".tmp"))
            os.replace(best_path.with_suffix(".tmp"), best_path)
        else:
            st["bad"] += 1
        print(f"epoch {st['epoch']:2d} loss {np.mean(losses):.4f} val MAE {val_mae_mw:7.1f} MW "
              f"({time.time() - t0:.0f}s)", flush=True)
        tmp = state_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "st": st}, tmp)
        os.replace(tmp, state_path)                                    # atomic: safe if killed
        last_epoch_s = time.time() - t0
    import pandas as pd
    pd.DataFrame(st["history"]).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    out = {}
    for split, s in [("val", va)] + ([("test", te)] if do_test else []):
        t1 = time.time()
        p_ = predict_dl(model, s)
        out[f"{split}_inference_s"] = time.time() - t1
        save_preds(run_dir / f"preds_{split}.npz", s, p_, scaler)
        d = np.load(run_dir / f"preds_{split}.npz")
        out[split] = all_metrics(d["y_true"], d["y_pred"])
    xp, xf, _ = TorchBatcher(te).make(torch.as_tensor(te.origins[:1]))
    with torch.no_grad():
        t1 = time.perf_counter()
        for _ in range(20):
            model(xp, xf)
        single_ms = (time.perf_counter() - t1) / 20 * 1000
    cost = {"params": int(sum(p.numel() for p in model.parameters())),
            "model_size_mb": best_path.stat().st_size / 1e6, "train_time_s": st["train_time"],
            "epochs": st["epoch"], "best_val_mae_mw": st["best"],
            "inference_ms_per_forecast": 1000 * out[("test" if do_test else "val") + "_inference_s"] /
            len(te if do_test else va), "inference_ms_single": single_ms,
            "peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "hardware": hardware()}
    try:
        state_path.unlink(missing_ok=True)                            # resume state no longer needed
    except OSError:
        pass                                                          # read-only/no-delete folders
    return out, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-test", action="store_true", help="tuning runs: never touch the test set")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    cfg = load_config(a.config, a.set)
    name = a.name or Path(a.config).stem
    run_dir = ROOT / "results" / "runs" / name / f"seed{a.seed}"
    if (run_dir / "metrics.json").exists():
        print(f"{run_dir} already done - skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump({**cfg, "seed": a.seed, "name": name, "test": not a.no_test}, open(run_dir / "config.json", "w"),
              indent=2)
    df = load_processed()
    scaler = json.load(open(ROOT / "data" / "processed" / "scaler.json"))
    srcs = make_sources(cfg, df)
    print(f"[{name} seed {a.seed}] train windows {len(srcs[0])}, val {len(srcs[1])}, test {len(srcs[2])}")
    runner = run_baseline if cfg["model"]["name"] in ("naive_week", "naive_day", "lightgbm") else run_deep
    args = (cfg, a.seed, run_dir, srcs, scaler, not a.no_test)
    out, cost = runner(*args) if runner is run_baseline else runner(*args, a.time_budget)
    if out is None:
        sys.exit(3)                                                    # unfinished (time budget)
    json.dump(out, open(run_dir / "metrics.json", "w"), indent=2)
    json.dump(cost, open(run_dir / "cost.json", "w"), indent=2)
    msg = f"val MAE {out['val']['MAE']:.1f} MW"
    if "test" in out:
        msg += f" | test MAE {out['test']['MAE']:.1f} MW, MAPE {out['test']['MAPE']:.2f}%"
    print(f"[{name} seed {a.seed}] done: {msg}")


if __name__ == "__main__":
    main()
