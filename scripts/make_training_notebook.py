"""Writes the Kaggle baseline training notebook (03) and its kernel metadata.

    python scripts/make_training_notebook.py <out_dir> <commit> <smoke: 0|1> [resume_dataset]

resume_dataset (optional): a Kaggle dataset holding last.pth / best.pth / history.csv from an earlier session, e.g.
darrensheehan/ecg-r1-checkpoint; it is attached and training resumes from it.
"""
import json
import sys
from pathlib import Path

out, commit, smoke = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
resume_dataset = sys.argv[4] if len(sys.argv) > 4 else None
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# 03 · Baseline training: Whole model, EfficientNet-B0, on two GPUs

Assignment 1, step 3: replicate the author's training script.

- **Code**: my fork of the author's training repository, at a fixed commit. Compared with the original it adds
  the resize fix (images now match the masks and the author's inference), two-GPU training
  (DistributedDataParallel, same total batch of 4), resuming from `last.pth`, a per-epoch `history.csv`,
  and a time budget so a Kaggle session ends cleanly before its 12-hour limit.
- **Setup**: the author's Whole-model config (EfficientNet-B0 U-Net, AdamW 1e-3, cosine schedule, BCE with
  pos_weight 20, fp16, gradient checkpointing). Trained on folds 1-4 (7,028 images) and validated on fold 0
  (1,764 images). No synthetic data.
- **Scale**: the author trained for 50 epochs. A measured 1.87 h per epoch on one T4 makes that about 94 GPU hours,
  so this run uses fewer epochs, and the cosine schedule is shortened to match.
- **Resuming**: if a `last.pth` from an earlier session is attached as an input, training continues from it.
""")

code(rf"""
SMOKE = {smoke}                  # True: a few minutes on a small sample, to test the two-GPU setup
EPOCHS = 12
MAX_HOURS = 11.0                 # stop after the epoch that would overrun this, leaving time to save outputs
REPO = "https://github.com/darren1sheehan/physionet-image-digitization-Assignment-1.git"
COMMIT = "{commit}"
OUT = "/kaggle/working/outputs/whole_b0_baseline"

!git clone -q $REPO /kaggle/working/repo && cd /kaggle/working/repo && git checkout -q $COMMIT && git log --oneline -1
!pip install -q "timm<1.0.29" "segmentation-models-pytorch==0.5.0" "albumentations==2.0.8" omegaconf
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os, torch
N_GPUS = torch.cuda.device_count()
print("GPUs:", N_GPUS, " CPUs:", os.cpu_count())
""")

code(r"""
# Locate the attached inputs (search rather than hard-code: Kaggle's mount paths have changed over time)
import glob, shutil
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
print(*rect.values(), mask_dir, comp_train, sep="\n")

# Fold CSV for the competition images, with paths pointing at the inputs
os.chdir("/kaggle/working/repo")
df = pd.read_csv("train_fold_with_synthesis.csv", dtype={"id": str, "type_id": str})
comp = df[~df["is_synthesis"]].copy()
comp["image_path"] = [str(rect[f] / Path(p).name) for f, p in zip(comp["fold"], comp["image_path"])]
if SMOKE:
    comp = pd.concat([comp[comp["fold"] != 0].sample(80, random_state=0), comp[comp["fold"] == 0].sample(40, random_state=0)])
missing = ~comp["image_path"].map(os.path.exists)
assert not missing.any(), f"{missing.sum()} images missing"
comp.to_csv("/kaggle/working/folds.csv", index=False)

# Ground truth for the validation fold: competition train/{id}/{id}.csv -> ecg_dir/{id}.csv
ecg_dir = Path("/kaggle/working/ecg_csv"); ecg_dir.mkdir(exist_ok=True)
for i in comp.loc[comp["fold"] == 0, "id"].unique():
    if not (ecg_dir / f"{i}.csv").exists():
        (ecg_dir / f"{i}.csv").symlink_to(comp_train / i / f"{i}.csv")
print("train:", (comp["fold"] != 0).sum(), " val:", (comp["fold"] == 0).sum())
""")

code(r"""
# Resume if an earlier session's last.pth is attached as an input; carry its history.csv over
os.makedirs(OUT, exist_ok=True)
previous = sorted(glob.glob("/kaggle/input/**/last.pth", recursive=True))  # from an attached earlier session
resume = []
if previous and not SMOKE:
    prev_dir = Path(previous[-1]).parent
    for name in ("last.pth", "best.pth", "history.csv"):
        if (prev_dir / name).exists():
            shutil.copy(prev_dir / name, OUT)
    resume = ["--resume", f"{OUT}/last.pth"]
    print("resuming from", prev_dir)
    print(open(f"{OUT}/history.csv").read())
else:
    print("starting from scratch")
""")

code(r"""
# Train (torchrun starts one process per GPU; training.batch_size=4 is the total over all GPUs)
epochs, hours = (1, 0.5) if SMOKE else (EPOCHS, MAX_HOURS)
args = ["--config", "configs/whole_model.yaml", "--output_dir", OUT, *resume,
        "data.csv_path=/kaggle/working/folds.csv", f"data.mask_dir={mask_dir}", f"data.ecg_dir={ecg_dir}",
        "wandb.enabled=false", f"training.num_epochs={epochs}", f"training.max_hours={hours}",
        f"num_workers={max(1, 4 // N_GPUS)}"]
launcher = ["torchrun", f"--nproc_per_node={N_GPUS}"] if N_GPUS > 1 else ["python"]
cmd = " ".join(launcher + ["train.py"] + args)
print(cmd)
!{cmd}
""")

code(r"""
# Training curves
import matplotlib.pyplot as plt
hist = pd.read_csv(f"{OUT}/history.csv")
print(hist.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist["epoch"], hist["train_loss"], "o-", label="train")
ax[0].plot(hist["epoch"], hist["val_loss"], "o-", label="validation")
ax[0].set(xlabel="epoch", ylabel="BCE loss", title="Loss"); ax[0].legend()
ax[1].plot(hist["epoch"], hist["val_snr"], "o-")
ax[1].set(xlabel="epoch", ylabel="dB", title="Validation SNR (fold 0, per-row proxy)")
plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=120); plt.show()
!ls -la $OUT
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
slug = "darren-sheehan-03-baseline-training-smoke" if smoke else "darren-sheehan-03-baseline-training"
title = "Darren-Sheehan-03-baseline-training" + ("-smoke" if smoke else "")
out.mkdir(parents=True, exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {
    "id": f"darrensheehan/{slug}", "title": title, "code_file": f"{slug}.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ([resume_dataset] if resume_dataset else []) +
                       [f"takashisomeya/physionet-rectified-images-fold{f}" for f in range(5)]
                       + ["takashisomeya/physionet-comp-syn-mask-x5000-500hz"],
    "competition_sources": ["physionet-ecg-image-digitization"],
    "kernel_sources": [], "model_sources": [],
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("written", out / f"{slug}.ipynb")
