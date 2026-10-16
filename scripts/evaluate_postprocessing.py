#!/usr/bin/env python
"""Offline evaluation of decoders (P2) and lead-consistency post-processing (P1 variants) with the competition metric.

Runs a trained Whole model over validation images (rectified pages) with the author's flip TTA, decodes each row's
probability map with several decoders, rebuilds the 12 leads as the author's make_submission does, applies each
post-processing variant and scores every combination with the competition metric against the ground truth.

    python scripts/evaluate_postprocessing.py --weights best.pth --folds-csv folds.csv --gt-dir ecg_csv \\
        --out results.csv [--limit 20] [--device cuda]

folds-csv needs the columns id, type_id, fold and image_path; only rows of --fold (default 0) are used.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.signal
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.transforms import resize_rectified  # noqa: E402
from src.evaluation.competition_metric import image_snr, score_db  # noqa: E402
from src.evaluation.postprocess import POSTPROCESS, decode  # noqa: E402
from src.models.whole_model import Net  # noqa: E402

OFFSET, T0, T1, MV_TO_PIXEL = 416, 301, 5301, 79.0
ZERO_MV = np.array([703.5, 987.5, 1271.5, 1531.5]) - OFFSET
ROWS = [['I', 'aVR', 'V1', 'V4'], ['II', 'aVL', 'V2', 'V5'], ['III', 'aVF', 'V3', 'V6']]
DECODERS = ["mean", "parabolic", "window"]
COMBOS = [("mean", "none"), ("mean", "p1"), ("mean", "p1b"), ("mean", "p1t"),
          ("parabolic", "none"), ("parabolic", "p1b"), ("window", "none"), ("window", "p1b")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", required=True)
    p.add_argument("--folds-csv", required=True)
    p.add_argument("--gt-dir", required=True, help="directory of {id}.csv ground-truth files")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--encoder", default="tu-timm/tf_efficientnet_b0.ns_jft_in1k")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="postprocessing_results.csv")
    return p.parse_args()


def load_model(path, encoder, device):
    model = Net(encoder_name=encoder, encoder_weights=None, whole_decoder_depth=4, loss_weight=20)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    result = model.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, result
    model.output_type = ["infer"]
    return model.to(device).eval()


@torch.no_grad()
def predict(model, image, device):
    """Author's flip TTA: average of the original and the horizontally flipped (and flipped back) prediction."""
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
    with torch.autocast(device_type=torch.device(device).type, enabled=device.startswith("cuda")):
        p = model({"image": x})["pixel"].float()[0]
        pf = model({"image": torch.flip(x, dims=[3])})["pixel"].float()[0]
    return 0.5 * (p + torch.flip(pf, dims=[2]))


def leads_from_rows(rows_mv, truth, sig_len):
    """Split the 4 decoded rows (each resampled to sig_len) into the 12 leads, as the author's make_submission."""
    leads = {}
    for r, names in enumerate(ROWS):
        lengths = [len(truth[k]) for k in names]
        if names[0] == 'II':
            lengths[0] = sig_len - sum(lengths[1:])
        for name, seg in zip(names, np.split(rows_mv[r], np.cumsum(lengths)[:-1])):
            leads[name] = seg
    ii_short = leads['II']
    leads['II'] = rows_mv[3]
    return leads, ii_short


def score_one(args):
    truth, fs, decoded = args
    out = {}
    for dec, post in COMBOS:
        leads, ii_short = decoded[dec]
        out[f"{dec}+{post}"] = image_snr(truth, POSTPROCESS[post](leads, ii_short), fs)
    return out


def main():
    args = parse_args()
    df = pd.read_csv(args.folds_csv, dtype={"id": str, "type_id": str})
    df = df[(df["fold"] == args.fold) & ~df.get("is_synthesis", False)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    model = load_model(args.weights, args.encoder, args.device)
    truths, jobs = {}, []
    t0 = time.time()
    for n, row in df.iterrows():
        if row["id"] not in truths:
            gt = pd.read_csv(Path(args.gt_dir) / f"{row['id']}.csv")
            truths[row["id"]] = {c: gt[c].dropna().values.astype(np.float64) for c in gt.columns}
        truth = truths[row["id"]]
        sig_len = len(truth['II'])
        image = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        image = resize_rectified(image)[OFFSET:1696, :5600]
        prob = predict(model, image, args.device)[:, :, T0:T1]
        decoded = {}
        for dec in DECODERS:
            y = decode(prob, dec).cpu().numpy()
            mv = (ZERO_MV[:, None] - y) / MV_TO_PIXEL
            rows = np.stack([scipy.signal.resample(s, sig_len) for s in mv])
            decoded[dec] = leads_from_rows(rows, truth, sig_len)
        jobs.append((truth, sig_len // 10, decoded))
        if n % 100 == 0:
            print(f"  inference {n}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    print(f"inference done: {len(jobs)} images in {time.time() - t0:.0f}s", flush=True)

    with Pool(args.workers) as pool:
        scores = pool.map(score_one, jobs, chunksize=8)
    res = pd.concat([df[["id", "type_id"]], pd.DataFrame(scores)], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    cols = [f"{d}+{p}" for d, p in COMBOS]
    summary = pd.DataFrame({"score_dB": [score_db(res[c]) for c in cols]}, index=cols)
    print("\nScore (competition metric, dB) on", len(res), "images:")
    print(summary.round(3).to_string())
    by_type = res.groupby("type_id")[cols].apply(lambda g: g.apply(score_db))
    print("\nBy image type (dB):")
    print(by_type.round(2).to_string())
    summary.to_csv(Path(args.out).with_name(Path(args.out).stem + "_summary.csv"))
    by_type.to_csv(Path(args.out).with_name(Path(args.out).stem + "_by_type.csv"))


if __name__ == "__main__":
    main()
