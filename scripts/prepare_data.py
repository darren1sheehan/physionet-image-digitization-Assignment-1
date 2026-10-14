#!/usr/bin/env python
"""Download the full training data onto a GPU machine (UCC server, Colab, cloud VM).

Replaces download_data.sh, whose last step fails because the author's ground-truth CSV
dataset is private. Ground truth comes from the competition files instead, and only for
the validation fold (training itself never reads it).

    data/rectified/{id}-{type}.rect.png        stage-1 rectified images, chosen folds
    data/mask_x5000_500hz_comp_synthesis/      sparse masks (all records + synthetic)
    data/kaggle_ecg_csv/{id}.csv               ground truth for the validation fold
    data/train_folds_available.csv             fold CSV restricted to what is on disk

Sizes: each image fold ~11.8 GB (needs ~2x free while unzipping), masks 2.1 GB,
each synthetic chunk ~6 GB (1000 images).

Usage:
    python scripts/prepare_data.py                          # all 5 folds, validate on fold 0
    python scripts/prepare_data.py --folds 0 1 2            # less data: train on 1-2, validate on 0
    python scripts/prepare_data.py --synthetic-chunks 2     # + 2000 wrinkled images

    python train.py --config configs/whole_model.yaml data.csv_path=data/train_folds_available.csv

Kaggle credentials: ~/.kaggle/access_token, ~/.kaggle/kaggle.json, or the KAGGLE_API_TOKEN /
KAGGLE_USERNAME+KAGGLE_KEY environment variables. The competition rules must be accepted.
"""
import argparse
import contextlib
import io
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "physionet-ecg-image-digitization"
RECT_DATASET = "takashisomeya/physionet-rectified-images-fold{fold}"
MASK_DATASET = "takashisomeya/physionet-comp-syn-mask-x5000-500hz"
SYN_DATASET = "takashisomeya/wrinkled-rect-images-{start:05d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Download the training data")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--synthetic-chunks", type=int, default=0, help="Wrinkled image chunks (1000 each, max 22)")
    parser.add_argument("--all-ground-truth", action="store_true", help="Ground truth for every record, not just val")
    parser.add_argument("--workers", type=int, default=8, help="Parallel ground-truth downloads")
    return parser.parse_args()


def download_dataset(api, dataset, dest: Path):
    """Download and unzip a whole Kaggle dataset into dest."""
    t = time.time()
    print(f"  {dataset} ...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        api.dataset_download_files(dataset, path=str(dest), unzip=True, quiet=True)
    print(f"  done in {(time.time() - t) / 60:.1f} min", flush=True)


def download_ground_truth(api, sample_ids, ecg_dir: Path, workers=8):
    """Fetch competition train/{id}/{id}.csv files in parallel; returns how many were downloaded."""
    todo = [i for i in sample_ids if not (ecg_dir / f"{i}.csv").exists()]
    ecg_dir.mkdir(parents=True, exist_ok=True)

    def fetch(sample_id):
        api.competition_download_file(COMPETITION, f"train/{sample_id}/{sample_id}.csv", path=str(ecg_dir), quiet=True)

    # redirect_stdout swaps a process-wide stream, so do it once here rather than inside each thread
    with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, todo))
    return len(todo)


def move_contents(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.move(str(f), dst / f.name)
    shutil.rmtree(src)


def main():
    args = parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    cfg = OmegaConf.load(ROOT / "configs" / "base.yaml")
    rect_dir = ROOT / cfg.data.rect_dir
    mask_dir = ROOT / cfg.data.mask_dir
    ecg_dir = ROOT / cfg.data.ecg_dir
    df = pd.read_csv(ROOT / cfg.data.csv_path, dtype={"id": str, "type_id": str})
    comp = df[~df["is_synthesis"]]
    tmp = ROOT / "data" / "_download"

    print("Rectified images")
    for fold in args.folds:
        rows = comp[comp["fold"] == fold]
        if all((ROOT / p).exists() for p in rows["image_path"]):
            print(f"  fold {fold}: already present ({len(rows)} images)")
            continue
        download_dataset(api, RECT_DATASET.format(fold=fold), tmp)
        move_contents(tmp / f"rectified_fold{fold}", rect_dir)

    for chunk in range(args.synthetic_chunks):
        download_dataset(api, SYN_DATASET.format(start=chunk * 1000), tmp)
        move_contents(tmp / "kaggle" / "working" / "outputs" / "rectified", rect_dir)
        shutil.rmtree(tmp, ignore_errors=True)

    print("Masks")
    if all((mask_dir / f"{i}.mask-coo.npz").exists() for i in comp["id"].unique()):
        print("  already present")
    else:
        download_dataset(api, MASK_DATASET, mask_dir.parent)

    print("Ground truth (competition files)")
    gt_ids = comp["id"] if args.all_ground_truth else comp.loc[comp["fold"] == args.val_fold, "id"]
    n = download_ground_truth(api, gt_ids.unique(), ecg_dir, args.workers)
    print(f"  downloaded {n} files")

    # Fold CSV with only the rows whose image and mask are on disk
    available = df[
        df["image_path"].map(lambda p: (ROOT / p).exists())
        & df["id"].map(lambda i: (mask_dir / f"{i}.mask-coo.npz").exists())
    ]
    out = ROOT / "data" / "train_folds_available.csv"
    available.to_csv(out, index=False)
    is_comp = ~available["is_synthesis"]
    print(f"\nWrote {out.relative_to(ROOT)}: "
          f"{(is_comp & (available['fold'] != args.val_fold)).sum()} train, "
          f"{(is_comp & (available['fold'] == args.val_fold)).sum()} val (fold {args.val_fold}), "
          f"{available['is_synthesis'].sum()} synthetic rows")


if __name__ == "__main__":
    main()
