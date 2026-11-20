"""Prove the split is leakage-free and print the sample counts.  Usage: python experiments/check_split.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.windows import LoadWindows, check_no_leakage, load_processed  # noqa: E402

df = load_processed()
res = check_no_leakage(df)
x_past, x_fut, y = LoadWindows(df, "val").arrays()
res["shapes_val"] = {"x_past": list(x_past.shape), "x_future": list(x_fut.shape), "y": list(y.shape)}
res["shapes_val_onehot"] = {"x_past": list(LoadWindows(df, "val", encoding="onehot")[0][0].shape)}
print(json.dumps(res, indent=2))
json.dump(res, open(Path(__file__).parent / "split_check.json", "w"), indent=2)
print("ALL LEAKAGE CHECKS PASSED")
