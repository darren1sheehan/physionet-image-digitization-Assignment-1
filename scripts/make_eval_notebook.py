#!/usr/bin/env python
"""Build the Kaggle notebook (08) that runs scripts/evaluate_postprocessing.py on fold 0 with a trained model.

    python scripts/make_eval_notebook.py <commit> <weights_kernel> <out_dir>
"""
import json
import sys
from pathlib import Path

commit, weights_kernel, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(f"""
# 08 · Offline evaluation of decoding (P2) and lead-consistency post-processing (P1 variants)

Assignment 1. Runs my trained Whole model (EfficientNet-B0, output of `{weights_kernel}`) over all of fold 0
(1,764 validation images, held out from training) with the author's flip TTA, and scores every combination of
decoder (`mean` = the author's, `parabolic` and `window` = 3rd place ideas) and post-processing (`none`, `p1` as
submitted in S1+P1, `p1b` offset-safe, `p1t` 1st place thresholds) with the competition metric against the ground
truth. No leaderboard submissions are used.
""")

code(rf"""
REPO = "https://github.com/darren1sheehan/physionet-image-digitization-Assignment-1.git"
COMMIT = "{commit}"
!git clone -q $REPO /kaggle/working/repo && cd /kaggle/working/repo && git checkout -q $COMMIT && git log --oneline -1
!pip install -q "timm<1.0.29" "segmentation-models-pytorch==0.5.0" "albumentations==2.0.8" omegaconf
""")

code(r"""
import glob, os
from pathlib import Path
import pandas as pd

def find_dir(name):
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        hits = [p for p in glob.glob(f"/kaggle/input/{depth}/{name}") if os.path.isdir(p)]
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(name)

rect0 = find_dir("rectified_fold0")
comp_train = find_dir("train")
weights = sorted(glob.glob("/kaggle/input/**/outputs/whole_b0_baseline/best.pth", recursive=True))
assert len(weights) == 1, weights
print("weights:", weights[0])

os.chdir("/kaggle/working/repo")
df = pd.read_csv("train_fold_with_synthesis.csv", dtype={"id": str, "type_id": str})
val = df[~df["is_synthesis"] & (df["fold"] == 0)].copy()
val["image_path"] = [str(rect0 / Path(p).name) for p in val["image_path"]]
val.to_csv("/kaggle/working/val_fold0.csv", index=False)
gt = Path("/kaggle/working/ecg_csv"); gt.mkdir(exist_ok=True)
for i in val["id"].unique():
    if not (gt / f"{i}.csv").exists():
        (gt / f"{i}.csv").symlink_to(comp_train / i / f"{i}.csv")
print(len(val), "validation images,", val["id"].nunique(), "records")
""")

code(r"""
!python scripts/evaluate_postprocessing.py --weights {weights[0]} --folds-csv /kaggle/working/val_fold0.csv \
    --gt-dir /kaggle/working/ecg_csv --workers 4 --out /kaggle/working/eval/results.csv
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
slug = "darren-sheehan-08-postprocessing-evaluation"
out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"darrensheehan/{slug}", "title": "Darren-Sheehan-08-postprocessing-evaluation", "code_file": f"{slug}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["takashisomeya/physionet-rectified-images-fold0"],
    "competition_sources": ["physionet-ecg-image-digitization"],
    "kernel_sources": [weights_kernel], "model_sources": [],
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("written", out / f"{slug}.ipynb")
