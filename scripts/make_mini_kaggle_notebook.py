#!/usr/bin/env python
"""Build a Kaggle notebook (07) that trains one mini-setup run for the training modifications.

Same setup as the Colab mini-baseline (notebooks/colab/06_modifications_colab.ipynb): Whole model, EfficientNet-B0,
the author's settings, R1's setup (train on folds 1-4, validate on all of fold 0, two GPUs, total batch 4) but
6 epochs instead of 12, seed 0. Only the modification differs. (A first version trained on folds 1-2 on one GPU;
its Colab pilot reached only about 1 dB validation SNR, too early in training for a meaningful comparison.)

    python scripts/make_mini_kaggle_notebook.py <commit> <run: mini_baseline|M1|M2|M3> <out_dir>
"""
import json
import sys
from pathlib import Path

commit, run, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OVERRIDES = {
    "mini_baseline": [],
    "M1": ["data.resize_mode=legacy"],
    "M2": ["data.grayscale_p=0.25"],
    "M3": ["model.snr_loss_weight=0.1"],
}
DESCRIPTION = {
    "mini_baseline": "no change (mini-baseline)",
    "M1": "training images resized as in the original repository (straight to 5600 px, 4-7 px drift from the masks); "
          "validation keeps the inference resize",
    "M2": "grayscale augmentation, `A.ToGray(p=0.25)`, as listed in the author's write-up but missing from his code",
    "M3": "SNR-aware loss on the decoded signal (weight 0.1) added to the author's BCE (idea from the 11th place solution)",
}
assert run in OVERRIDES, run
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(f"""
# 07 · Training modification {run} (mini-setup)

Assignment 1. The author's Whole model (EfficientNet-B0, his settings) trained in a small, fixed setup with **one**
change: **{DESCRIPTION[run]}**.

Setup, the same for every run: R1's setup (notebook 03): train on folds 1-4 (7,028 images), validate on all of
fold 0 (1,764 images), two GPUs with a total batch of 4, seed 0, but 6 epochs (cosine schedule over 6) instead of 12.
The validation SNR of this run is compared with the mini-baseline trained the same way.
""")

code(rf"""
RUN = "{run}"
OVERRIDES = {json.dumps(OVERRIDES[run])}
EPOCHS = 6
REPO = "https://github.com/darren1sheehan/physionet-image-digitization-Assignment-1.git"
COMMIT = "{commit}"
OUT = f"/kaggle/working/outputs/whole_b0_{{RUN.lower()}}"

!git clone -q $REPO /kaggle/working/repo && cd /kaggle/working/repo && git checkout -q $COMMIT && git log --oneline -1
!pip install -q "timm<1.0.29" "segmentation-models-pytorch==0.5.0" "albumentations==2.0.8" omegaconf
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code(r"""
# Locate the attached inputs, build the mini fold CSV and link the validation ground truth
import glob, os
from pathlib import Path
import pandas as pd

def find_dir(name):
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        hits = [p for p in glob.glob(f"/kaggle/input/{depth}/{name}") if os.path.isdir(p)]
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(name)

rect = {f: find_dir(f"rectified_fold{f}") for f in range(5)}
mask_dir = find_dir("mask_x5000_500hz_comp_synthesis")
comp_train = find_dir("train")

os.chdir("/kaggle/working/repo")
df = pd.read_csv("train_fold_with_synthesis.csv", dtype={"id": str, "type_id": str})
mini = df[~df["is_synthesis"]].copy()
mini["image_path"] = [str(rect[f] / Path(p).name) for f, p in zip(mini["fold"], mini["image_path"])]
assert mini["image_path"].map(os.path.exists).all()
mini.to_csv("/kaggle/working/mini_folds.csv", index=False)

ecg_dir = Path("/kaggle/working/ecg_csv"); ecg_dir.mkdir(exist_ok=True)
for i in mini.loc[mini["fold"] == 0, "id"].unique():
    if not (ecg_dir / f"{i}.csv").exists():
        (ecg_dir / f"{i}.csv").symlink_to(comp_train / i / f"{i}.csv")
print("train:", (mini["fold"] != 0).sum(), " val:", (mini["fold"] == 0).sum())
""")

code(r"""
# Train on both GPUs, as R1 (torchrun; training.batch_size=4 is the total over the GPUs)
os.makedirs(OUT, exist_ok=True)
args = ["--config", "configs/whole_model.yaml", "--output_dir", OUT,
        "data.csv_path=/kaggle/working/mini_folds.csv", f"data.mask_dir={mask_dir}", f"data.ecg_dir={ecg_dir}",
        "cv.val_fold=0", "wandb.enabled=false", f"training.num_epochs={EPOCHS}", "training.max_hours=11.0",
        "num_workers=2", *OVERRIDES]
cmd = " ".join(["torchrun", "--nproc_per_node=2", "train.py"] + args)
print(cmd)
!{cmd}
""")

code(r"""
# Curves and summary
import matplotlib.pyplot as plt
hist = pd.read_csv(f"{OUT}/history.csv")
print(RUN, OVERRIDES); print(hist.to_string(index=False))
print(f"best val SNR: {hist['val_snr'].max():.2f} dB (epoch {int(hist.loc[hist['val_snr'].idxmax(), 'epoch'])})")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist["epoch"], hist["train_loss"], "o-", label="train"); ax[0].plot(hist["epoch"], hist["val_loss"], "o-", label="val")
ax[0].set(xlabel="epoch", ylabel="loss", title=f"{RUN}: loss"); ax[0].legend()
ax[1].plot(hist["epoch"], hist["val_snr"], "o-"); ax[1].set(xlabel="epoch", ylabel="dB", title=f"{RUN}: validation SNR (fold 0)")
plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=120); plt.show()
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
slug = "darren-sheehan-07-mini-baseline" if run == "mini_baseline" else f"darren-sheehan-07-mini-{run.lower()}"
out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"darrensheehan/{slug}", "title": slug.replace("darren-sheehan-", "Darren-Sheehan-"),
    "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [f"takashisomeya/physionet-rectified-images-fold{f}" for f in range(5)]
                       + ["takashisomeya/physionet-comp-syn-mask-x5000-500hz"],
    "competition_sources": ["physionet-ecg-image-digitization"], "kernel_sources": [], "model_sources": [],
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("written", out / f"{slug}.ipynb")
