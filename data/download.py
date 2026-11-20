"""Download the PJM East hourly load file and verify its checksum.

Usage:  python data/download.py

Primary (documented) source : Kaggle dataset robikscube/hourly-energy-consumption
                              (needs a Kaggle account: `kaggle datasets download -d robikscube/hourly-energy-consumption`)
Mirror used by this script  : GitHub panambY/Hourly_Energy_Consumption (same file, no login),
                              pinned by SHA-256 so everyone in the team gets the identical version.
"""
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["dataset"]


def fetch(url: str, out: Path) -> None:
    """urllib first; fall back to curl (macOS Python often lacks root certificates)."""
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:  # noqa: BLE001
        if not shutil.which("curl"):
            raise
        print(f"  urllib failed ({type(e).__name__}), retrying with curl")
        subprocess.run(["curl", "-sSL", "-o", str(out), url], check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_regions() -> None:
    """Other PJM regions, used only for pretraining (src/datasets/regions.py)."""
    sys.path.insert(0, str(ROOT))
    from src.datasets.regions import BASE_URL, REGIONS, region_path
    for r in REGIONS:
        p = region_path(r)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            print(f"Downloading {r} ...")
            fetch(BASE_URL.format(r), p)
    print(f"OK  {len(REGIONS)} pretraining regions in {p.parent}")


def main() -> None:
    out = ROOT / CFG["raw_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        print(f"Downloading {CFG['url']} ...")
        fetch(CFG["url"], out)
    digest = sha256(out)
    if digest != CFG["expected_sha256"]:
        sys.exit(f"Checksum mismatch!\n got      {digest}\n expected {CFG['expected_sha256']}")
    print(f"OK  {out}  sha256={digest[:16]}...")
    download_regions()


if __name__ == "__main__":
    main()
