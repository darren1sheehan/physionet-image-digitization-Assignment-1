#!/usr/bin/env python
"""Download a small, realistic subset of the training data for local work.

The full training data is ~61 GB (five folds of rectified images plus masks),
far more than is needed to explore the data or smoke-test train.py on a laptop.
This script downloads individual files for a handful of records instead, and
writes a matching fold CSV that train.py can use directly:

    data/rectified/{id}-{type}.rect.png             stage-1 rectified images
    data/mask_x5000_500hz_comp_synthesis/{id}...npz sparse 500 Hz masks
    data/kaggle_ecg_csv/{id}.csv                    ground-truth signals (*)
    data/competition/train/{id}/{id}-{type}.png     original images (--originals) (*)
    data/sample_folds.csv                           rows of the fold CSV that are present

(*) The author's flattened ECG CSV dataset (takashisomeya/physionet-ecg-kaggle-data-csv)
    is private, so ground truth comes from the competition files instead. Those need
    the competition rules accepted once on kaggle.com.

Usage:
    python scripts/download_sample.py                    # 2 val + 2 train records, all image types
    python scripts/download_sample.py --n-val 3 --n-train 5
    python scripts/download_sample.py --originals        # also the raw competition images
    python scripts/download_sample.py --n-synthetic 4    # also a few wrinkled (0015) images
    python scripts/download_sample.py --types 0001 --out data/smoke_folds.csv   # tiny CSV, no new downloads

    python train.py --config configs/whole_model.yaml data.csv_path=data/sample_folds.csv
"""
import argparse
import contextlib
import io
import zipfile
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "physionet-ecg-image-digitization"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
RECT_DATASET = "takashisomeya/physionet-rectified-images-fold{fold}"
MASK_DATASET = "takashisomeya/physionet-comp-syn-mask-x5000-500hz"
# Synthetic ids 00001-00999 all live in the first chunk of the wrinkled image datasets
SYN_DATASET = "takashisomeya/wrinkled-rect-images-00000"


def parse_args():
    parser = argparse.ArgumentParser(description="Download a small data sample")
    parser.add_argument("--n-val", type=int, default=2, help="Records from the validation fold")
    parser.add_argument("--n-train", type=int, default=2, help="Records from the other folds")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--types", nargs="*", default=None, help="Image types, e.g. 0001 0006 (default: all)")
    parser.add_argument("--n-synthetic", type=int, default=0, help="Wrinkled synthetic (0015) images")
    parser.add_argument("--originals", action="store_true", help="Also fetch the unrectified competition images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/sample_folds.csv", help="Fold CSV listing the selected rows")
    return parser.parse_args()


def fetch(download, src: str, dest_dir: Path) -> Path:
    """Run `download` unless dest_dir already holds the file; unzip if it arrives zipped."""
    name = src.rsplit("/", 1)[-1]
    dest = dest_dir / name
    if not dest.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):  # the API prints a URL per file even when quiet
            download()
        zipped = dest_dir / f"{name}.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(dest_dir)
            zipped.unlink()
    return dest


def main():
    args = parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    def dataset_file(dataset, src, dest_dir):
        return fetch(lambda: api.dataset_download_file(dataset, src, path=str(dest_dir), quiet=True), src, dest_dir)

    def competition_file(src, dest_dir):
        return fetch(lambda: api.competition_download_file(COMPETITION, src, path=str(dest_dir), quiet=True), src, dest_dir)

    cfg = OmegaConf.load(ROOT / "configs" / "base.yaml")
    mask_dir = ROOT / cfg.data.mask_dir
    ecg_dir = ROOT / cfg.data.ecg_dir
    df = pd.read_csv(ROOT / cfg.data.csv_path, dtype={"id": str, "type_id": str})

    # Pick whole records (all image types of an id) so train/val never share an ECG
    comp = df[~df["is_synthesis"]]
    val_ids = comp.loc[comp["fold"] == args.val_fold, "id"].drop_duplicates()
    train_ids = comp.loc[comp["fold"] != args.val_fold, "id"].drop_duplicates()
    ids = pd.concat([
        val_ids.sample(args.n_val, random_state=args.seed),
        train_ids.sample(args.n_train, random_state=args.seed),
    ]).tolist()
    rows = comp[comp["id"].isin(ids)]
    if args.types:
        rows = rows[rows["type_id"].isin(args.types)]
    syn_rows = df[df["is_synthesis"] & (df["id"].astype(int) < 1000)].head(args.n_synthetic)

    print(f"Records: {ids} (first {args.n_val} from val fold {args.val_fold})")

    print(f"Rectified images: {len(rows)} competition + {len(syn_rows)} synthetic")
    for row in rows.itertuples():
        name = Path(row.image_path).name
        dataset_file(RECT_DATASET.format(fold=row.fold), f"rectified_fold{row.fold}/{name}", ROOT / Path(row.image_path).parent)
    for row in syn_rows.itertuples():
        name = Path(row.image_path).name
        dataset_file(SYN_DATASET, f"kaggle/working/outputs/rectified/{name}", ROOT / Path(row.image_path).parent)

    print(f"Masks: {len(ids) + len(syn_rows)}")
    for sample_id in ids + syn_rows["id"].tolist():
        dataset_file(MASK_DATASET, f"{mask_dir.name}/{sample_id}.mask-coo.npz", mask_dir)

    print(f"Ground-truth signals{' and original images' if args.originals else ''} (competition data)")
    try:
        for sample_id in ids:
            competition_file(f"train/{sample_id}/{sample_id}.csv", ecg_dir)
            if args.originals:
                for type_id in rows.loc[rows["id"] == sample_id, "type_id"]:
                    competition_file(
                        f"train/{sample_id}/{sample_id}-{type_id}.png",
                        ROOT / "data" / "competition" / "train" / sample_id,
                    )
    except Exception as e:  # kagglesdk raises requests' HTTPError
        if "403" not in str(e):
            raise
        print(f"  403 Forbidden: accept the competition rules at {RULES_URL} and re-run this script.")
        print("  (Training still works without these; only the validation SNR needs them.)")

    # Fold CSV of the selected rows whose image and mask made it to disk
    selected = pd.concat([rows, syn_rows])
    present = selected[
        selected["image_path"].map(lambda p: (ROOT / p).exists())
        & selected["id"].map(lambda i: (mask_dir / f"{i}.mask-coo.npz").exists())
    ]
    out = ROOT / args.out
    present.to_csv(out, index=False)

    is_val = ~present["is_synthesis"] & (present["fold"] == args.val_fold)
    n_syn = present["is_synthesis"].sum()
    comp_ids = present.loc[~present["is_synthesis"], "id"].unique()
    n_truth = sum((ecg_dir / f"{i}.csv").exists() for i in comp_ids)
    print(f"Wrote {out.relative_to(ROOT)}: {len(present) - is_val.sum() - n_syn} train, {is_val.sum()} val "
          f"(val_fold={args.val_fold}), {n_syn} synthetic rows; ground truth for {n_truth}/{len(comp_ids)} records")


if __name__ == "__main__":
    main()
