"""Run every experiment in experiments/plan.yaml that is not finished yet.

  python experiments/run_plan.py --workers 4 --threads 1    # 4 parallel 1-thread jobs (recommended)
  python experiments/run_plan.py --workers 4 --budget 165   # stop after ~165 s
  python experiments/run_plan.py --only lstm tft            # a subset
  python experiments/run_plan.py --status                   # what is done / in progress / waiting

Progress is printed live (job start / finish / latest epoch every --every seconds); the full output of
each job goes to results/runs/<name>/seed<k>/train.log.
Jobs are resumable (src/train.py saves state after every epoch), so a stopped run continues later.
Ctrl-C stops every job cleanly; run the same command again to continue.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _ensure_env():
    """Re-launch with the project's .venv interpreter if the current one lacks the dependencies
    (e.g. `python3 experiments/run_plan.py` typed without activating the venv)."""
    try:
        import numpy, pandas, torch, yaml  # noqa: F401,E401
        return
    except ImportError as e:
        missing = e.name
    in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    if VENV_PY.exists() and not in_venv and not os.environ.get("RUN_PLAN_REEXEC"):
        print(f"[run_plan] '{missing}' is not installed for {sys.executable}; "
              f"switching to {VENV_PY.relative_to(ROOT)}", flush=True)
        os.environ["RUN_PLAN_REEXEC"] = "1"
        os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])
    sys.exit(f"[run_plan] missing package '{missing}'. Run:\n"
             f"  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")


_ensure_env()
import yaml  # noqa: E402

# rough relative cost per epoch, used to start the slowest jobs first (shorter total wall-clock)
COST = {"lstm": 3.0, "tft": 3.0, "transformer": 1.0, "lightgbm": 0.5}


def jobs(plan, only=None):
    out = []
    for r in plan["runs"]:
        if only and r["name"] not in only:
            continue
        for s in r.get("seeds", plan["seeds"]):
            out.append({**r, "seed": s})
    return out


def run_dir(j):
    return ROOT / "results" / "runs" / j["name"] / f"seed{j['seed']}"


def tag(j):
    return f"{j['name']} seed{j['seed']}"


def done(j):
    return (run_dir(j) / "metrics.json").exists()


def ready(j, plan):
    dep = j.get("after")
    return dep is None or all(done(x) for x in jobs(plan, [dep]))


def est_cost(j):
    cfg = yaml.safe_load(open(ROOT / j["config"]))
    model = cfg["model"]["name"]
    epochs = cfg.get("train", {}).get("max_epochs", 60)
    for s in j.get("set") or []:
        if str(s).startswith("train.max_epochs="):
            epochs = int(str(s).split("=", 1)[1])
        if str(s).startswith("data.lookback="):
            epochs *= int(str(s).split("=", 1)[1]) / 168
    return COST.get(model, 0.1) * epochs


def cmd(j, budget, threads):
    c = [sys.executable, "-u", str(ROOT / "src" / "train.py"), "--config", j["config"],
         "--seed", str(j["seed"]), "--name", j["name"]]
    if j.get("set"):
        c += ["--set", *[str(x) for x in j["set"]]]
    if j.get("test") is False:
        c.append("--no-test")
    if budget:
        c += ["--time-budget", str(budget)]
    if threads:
        c += ["--threads", str(threads)]
    return c


def last_line(j, prefix=("epoch", "[", "resumed")):
    try:
        lines = (run_dir(j) / "train.log").read_text(errors="replace").splitlines()
    except OSError:
        return ""
    for ln in reversed(lines):
        if ln.startswith(prefix):
            return ln.strip()
    return ""


def fmt(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="torch threads per job")
    ap.add_argument("--budget", type=float, default=None, help="wall-clock seconds for this call")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--every", type=float, default=60, help="seconds between progress reports")
    a = ap.parse_args()
    plan = yaml.safe_load(open(ROOT / "experiments" / "plan.yaml"))
    js = jobs(plan, a.only)
    if a.status:
        for j in js:
            st = "done" if done(j) else ("partial" if (run_dir(j) / "state.pt").exists() else "todo")
            print(f"{st:8s} {tag(j)}")
        print(f"{sum(done(j) for j in js)}/{len(js)} done")
        return

    pending = sorted([j for j in js if not done(j)], key=est_cost, reverse=True)
    n_done0 = len(js) - len(pending)
    print(f"[run_plan] {n_done0}/{len(js)} runs already done, {len(pending)} to run "
          f"with {a.workers} workers x {a.threads or 'default'} threads  (python: {sys.executable})", flush=True)
    if not pending:
        print("[run_plan] nothing to do - next: python src/evaluate.py")
        return
    print("[run_plan] per-job logs: results/runs/<name>/seed<k>/train.log   (Ctrl-C to stop, re-run to resume)",
          flush=True)

    t0, last_report, failed = time.time(), time.time(), []
    running = []   # (job, process, start_time)
    interrupted = False

    def stop_all(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    while (pending or running) and not interrupted:
        remaining = None if a.budget is None else a.budget - (time.time() - t0)
        still = []
        for j, p, ts in running:
            rc = p.poll()
            if rc is None:
                still.append((j, p, ts))
            elif rc == 0 and done(j):
                print(f"  done   {tag(j):32s} {fmt(time.time() - ts):>7s}  {last_line(j, ('[',))}", flush=True)
            elif rc == 3:
                print(f"  paused {tag(j):32s} (time budget) - re-run to resume", flush=True)
            else:
                failed.append(j)
                print(f"  FAILED {tag(j):32s} exit code {rc}; last lines of {run_dir(j).relative_to(ROOT)}/train.log:",
                      flush=True)
                tail = (run_dir(j) / "train.log").read_text(errors="replace").splitlines()[-8:]
                print("    " + "\n    ".join(tail), flush=True)
        running = still
        if remaining is not None and remaining < 30:
            break
        while len(running) < a.workers:
            nxt = next((j for j in pending if ready(j, plan)), None)
            if nxt is None:
                break
            pending.remove(nxt)
            log = run_dir(nxt)
            log.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if a.threads:   # also cap BLAS/OpenMP threads so parallel jobs do not oversubscribe the CPU
                env.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
                           VECLIB_MAXIMUM_THREADS=str(a.threads))
            p = subprocess.Popen(cmd(nxt, None if remaining is None else remaining - 10, a.threads),
                                 cwd=ROOT, stdout=open(log / "train.log", "a"), stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)   # Ctrl-C is handled here, not in the child
            running.append((nxt, p, time.time()))
            print(f"  start  {tag(nxt)}", flush=True)
        if not running:
            break
        if time.time() - last_report >= a.every:
            last_report = time.time()
            n = sum(done(j) for j in js)
            print(f"[{fmt(time.time() - t0)}] {n}/{len(js)} done, {len(running)} running, "
                  f"{len(pending)} waiting", flush=True)
            for j, _, ts in running:
                print(f"    {tag(j):32s} {fmt(time.time() - ts):>7s}  {last_line(j)}", flush=True)
        time.sleep(2)

    if interrupted:
        print("\n[run_plan] stopping - progress is saved after every epoch; run the same command to resume",
              flush=True)
        for _, p, _ in running:
            p.terminate()
    for _, p, _ in running:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    n = sum(done(j) for j in js)
    print(f"[run_plan] {n}/{len(js)} done in plan ({fmt(time.time() - t0)} this call)", flush=True)
    if failed:
        print(f"[run_plan] {len(failed)} failed: " + ", ".join(tag(j) for j in failed))
        sys.exit(1)
    if n == len(js):
        print("[run_plan] all runs finished - next: python src/evaluate.py && python experiments/error_analysis.py")


if __name__ == "__main__":
    main()
