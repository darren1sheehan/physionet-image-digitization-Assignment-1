#!/usr/bin/env python
"""Build the Colab notebook that trains the mini-setup runs for the training modifications (M1-M3).

    python scripts/make_colab_notebook.py <commit> notebooks/colab/06_modifications_colab.ipynb
"""
import json
import sys
from pathlib import Path

commit, out = sys.argv[1], Path(sys.argv[2])
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# 06 · Training modifications on Colab (mini-setup)

Assignment 1. Each run trains the author's Whole model (EfficientNet-B0, his settings) in the same small setup and
changes **one** thing, so differences in validation SNR can be attributed to it:

| RUN | Change |
|---|---|
| `mini_baseline` | none |
| `M1` | training images resized as in the original repository (straight to 5600 px, 4-7 px drift from the masks) |
| `M2` | grayscale augmentation, `A.ToGray(p=0.25)` as listed in the author's write-up but missing from his code |
| `M3` | SNR-aware loss on the decoded signal, weight 0.1, added to his BCE (idea from the 11th place solution) |

Mini-setup: train on folds 1-2 (about 3,500 images), validate on all of fold 0 (1,764 images), 6 epochs, seed 0.

**Before running**: Runtime -> Change runtime type -> **T4 GPU** (or L4). Add your Kaggle API token as a Colab secret
named `KAGGLE_API_TOKEN` (key icon in the left bar), or paste it when asked. Checkpoints, `history.csv` and the curves
go to Google Drive if it can be mounted, and at the end to a private Kaggle dataset `darrensheehan/ecg-colab-<run>`
(works in the VS Code Colab extension too). A disconnected session resumes: run all cells again with the same `RUN`.
In VS Code the token is asked for in a password box if Colab secrets are not available.
""")

code(rf"""
RUN = "mini_baseline"            # mini_baseline | M1 | M2 | M3
EPOCHS = 6
TRAIN_FOLDS, VAL_FOLD = [1, 2], 0
MAX_HOURS = 20                   # stop cleanly before Colab's session limit; rerun to resume

OVERRIDES = {{
    "mini_baseline": [],
    "M1": ["data.resize_mode=legacy"],
    "M2": ["data.grayscale_p=0.25"],
    "M3": ["model.snr_loss_weight=0.1"],
}}[RUN]
REPO = "https://github.com/darren1sheehan/physionet-image-digitization-Assignment-1.git"
COMMIT = "{commit}"
""")

code(r"""
# Output folder on Google Drive (falls back to the local disk if Drive cannot be mounted)
import os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = f"/content/drive/MyDrive/ecg_assignment1/{RUN}"
except Exception as e:
    print("Drive not mounted:", e, "- saving to /content/outputs (download it before the session ends)")
    OUT = f"/content/outputs/{RUN}"
os.makedirs(OUT, exist_ok=True)
print("outputs ->", OUT)

# Kaggle token: Colab secret KAGGLE_API_TOKEN, or paste it
try:
    from google.colab import userdata
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
except Exception:
    from getpass import getpass
    os.environ["KAGGLE_API_TOKEN"] = getpass("Kaggle API token: ")
""")

code(r"""
# Code at a fixed commit, and the pinned libraries
!rm -rf /content/repo && git clone -q $REPO /content/repo && cd /content/repo && git checkout -q $COMMIT && git log --oneline -1
!pip install -q "timm<1.0.29" "segmentation-models-pytorch==0.5.0" "albumentations==2.0.8" omegaconf "kaggle==1.8.3"
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code(r"""
# Data: rectified images for the folds used, sparse masks, and ground truth for the validation fold (~38 GB, 10-20 min)
%cd /content/repo
folds = " ".join(str(f) for f in sorted(set(TRAIN_FOLDS + [VAL_FOLD])))
!python scripts/prepare_data.py --folds {folds} --val-fold {VAL_FOLD}
!df -h /content | tail -1
""")

code(r"""
# Fold CSV for the mini-setup
import pandas as pd
df = pd.read_csv("data/train_folds_available.csv", dtype={"id": str, "type_id": str})
mini = df[~df["is_synthesis"] & df["fold"].isin(TRAIN_FOLDS + [VAL_FOLD])]
mini.to_csv("data/mini_folds.csv", index=False)
print("train:", (mini["fold"] != VAL_FOLD).sum(), " val:", (mini["fold"] == VAL_FOLD).sum())
""")

code(r"""
# Results are also kept in a private Kaggle dataset (works without Google Drive). If it already holds a
# last.pth from an earlier session and OUT does not, fetch it so training resumes.
import subprocess
DS = "darrensheehan/ecg-colab-" + RUN.lower().replace("_", "-")
if not os.path.exists(f"{OUT}/last.pth"):
    r = subprocess.run(["kaggle", "datasets", "download", "-d", DS, "-p", OUT, "--unzip"], capture_output=True, text=True)
    print("resume data from", DS, ":", "found" if os.path.exists(f"{OUT}/last.pth") else "none (starting from scratch)")
""")

code(r"""
# Train (resumes from OUT/last.pth if an earlier session got that far)
resume = ["--resume", f"{OUT}/last.pth"] if os.path.exists(f"{OUT}/last.pth") else []
args = ["--config", "configs/whole_model.yaml", "--output_dir", OUT, *resume,
        "data.csv_path=data/mini_folds.csv", f"cv.val_fold={VAL_FOLD}", "wandb.enabled=false",
        f"training.num_epochs={EPOCHS}", f"training.max_hours={MAX_HOURS}", "num_workers=2", *OVERRIDES]
cmd = " ".join(["python", "train.py"] + args)
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
ax[1].plot(hist["epoch"], hist["val_snr"], "o-"); ax[1].set(xlabel="epoch", ylabel="dB", title=f"{RUN}: validation SNR (fold {VAL_FOLD})")
plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=120); plt.show()
""")

code(r"""
# Save the results to the private Kaggle dataset DS (new version if it exists). The weights can then be attached
# to a Kaggle submission notebook, and a later session can resume from last.pth.
import json, shutil
up = "/content/upload"
shutil.rmtree(up, ignore_errors=True); os.makedirs(up)
for f in ("best.pth", "last.pth", "history.csv", "config.yaml", "curves.png"):
    if os.path.exists(f"{OUT}/{f}"):
        shutil.copy(f"{OUT}/{f}", up)
json.dump({"title": f"ECG assignment Colab run {RUN}", "id": DS, "licenses": [{"name": "other"}]},
          open(f"{up}/dataset-metadata.json", "w"))
exists = subprocess.run(["kaggle", "datasets", "status", DS], capture_output=True, text=True).returncode == 0
cmd = (["kaggle", "datasets", "version", "-p", up, "-m", f"{RUN}: {len(hist)} epochs"] if exists
       else ["kaggle", "datasets", "create", "-p", up])
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
print("results ->", f"https://www.kaggle.com/datasets/{DS}")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 4}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("written", out)
