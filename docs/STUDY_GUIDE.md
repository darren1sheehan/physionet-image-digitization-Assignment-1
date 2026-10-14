# Study guide: PhysioNet ECG Digitization, 2nd place (Takashi Someya)

Goal of the assignment: understand this top-10 solution and recreate its training.

| Resource | Link |
|---|---|
| Write-up | https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution |
| Training code (this fork's `upstream`) | https://github.com/someya-takashi/physionet-image-digitization |
| Inference notebook (6-model ensemble) | https://www.kaggle.com/code/takashisomeya/physionet-2nd-place-submission |
| Stages 0-1, reused unchanged | https://www.kaggle.com/code/hengck23/demo-submission |
| Metric discussion | https://www.kaggle.com/competitions/physionet-ecg-image-digitization/discussion/663901 |

Both notebooks are saved in `reference/` (gitignored, read-only copies).

## 1. Environment

- Conda env `physionet-ecg`: Python 3.12, PyTorch CPU build, smp 0.5, timm 1.0.28, albumentations 2.0.8.
  - VS Code terminals and notebooks pick it up from `.vscode/settings.json`.
  - Elsewhere: `conda activate physionet-ecg` in an Anaconda Prompt, or call
    `C:\Users\darre\anaconda3\envs\physionet-ecg\python.exe` directly.
  - Recreate on any machine: `conda env create -f environment.yml`.
- This laptop has no NVIDIA GPU, so local runs are for reading, debugging and smoke tests only.
  - Measured: Whole model (EfficientNet-B0, fp32, batch 1) takes ~71 s per training image and ~13 s per validation
    image, with ~5 GB peak RAM.
  - At that rate one epoch over ~7,000 training images would take about 6 days. Real training needs a GPU.
- Kaggle CLI 1.8.3 authenticates through `~/.kaggle/access_token` (user `darrensheehan`).
  Its "outdated version" warning is harmless.

## 2. Data

| What | Kaggle source | Size | Local path |
|---|---|---|---|
| Rectified images (stage-1 output), 5 folds | `takashisomeya/physionet-rectified-images-fold{0..4}` | 11.8 GB per fold, ~59 GB total | `data/rectified/` |
| Sparse masks (competition + synthetic) | `takashisomeya/physionet-comp-syn-mask-x5000-500hz` | 2.1 GB | `data/mask_x5000_500hz_comp_synthesis/` |
| Ground-truth signals | competition files `train/{id}/{id}.csv` | small | `data/kaggle_ecg_csv/` |
| Synthetic wrinkled images (optional) | `takashisomeya/wrinkled-rect-images-{00000..21000}` | 22 × ~6 GB | `data/rectified/` |
| Author's final weights | `takashisomeya/physionet-final-submission-models` | 3.0 GB | n/a |

- `train_fold_with_synthesis.csv` lists 977 ECG records × 9 image types (8,792 rows), split into 5 folds by record,
  plus 20,822 synthetic rows (`fold=-1`).
- `python scripts/download_sample.py` fetches individual files for a few records (~250 MB) and writes
  `data/sample_folds.csv`, which `train.py` accepts via `data.csv_path=...`.
- The ground-truth CSVs come from the competition, whose rules are now accepted. The script fetches them automatically.

## 3. The pipeline at a glance

```
photo / scan
  -> Stage 0 (hengck23): keypoints + homography      -> "normalised" page (rotation and perspective fixed)
  -> Stage 1 (hengck23): grid-point detection + warp  -> "rectified" page, 1700 x 2200 px (letter @ 200 dpi)
  -> Stage 2 (this repo): segmentation               -> 4 probability maps, one per ECG row
  -> post-processing: weighted mean y per column -> resample to the record's length -> mV -> 12 leads
```

The author's work is Stage 2 plus post-processing. The training input is the precomputed stage-1 output.

Geometry shared by configs, datasets and inference:

- The rectified page is stretched horizontally so that 10 s at 500 Hz is exactly 5,000 px: columns 301-5300 of a 5,600-wide image.
- Row baselines (0 mV) sit at y = 703.5, 987.5, 1271.5 and 1531.5. Scale is 79 px per mV (10 mm/mV at 200 dpi).
- Rows: `[I, aVR, V1, V4]`, `[II, aVL, V2, V5]`, `[III, aVF, V3, V6]`, `[II rhythm strip, full 10 s]`.

## 4. Step-by-step plan

1. **From photo to training data.** Notebook: `notebooks/01_from_photo_to_training_data.ipynb`.
   - Runs hengck23's Stage 0 (lead-label keypoints, then a homography) and Stage 1 (grid detection, then
     `grid_sample` un-warp) live on a raw photo.
   - Its output matches the author's rectified dataset (mean difference ~2/255).
   - Then covers the rectified geometry, the 12 leads to 4 rows mapping, and the sparse masks.
   - Ends by rebuilding the signal from a mask: ~75-90 dB SNR for 500 Hz records, ~40-50 dB for 256/512 Hz ones.

   Key ideas:
   - The competition signals are PTB-XL resampled to 250-1025 Hz. Masks are drawn from the original 500 Hz signals
     (single-model LB went from 21.67 to 22.49 dB).
   - Masks are sparse and sub-pixel: at most 2 px per column, with weights summing to 1.
2. **Datasets and augmentation.** Read `src/data/dataset_whole.py`, `dataset_series.py` and `transforms.py`.
   - Whole model: rows 416-1695, giving a 1280 × 5600 input.
   - Series model: four 480 px windows (±240 px around each baseline). Labels outside a window are clamped to its edge.
   - Horizontal flip is applied to image and mask together. `data.num_synthesis_data` adds synthetic placeholder rows.
3. **Models.** Read `src/models/whole_model.py` and `series_model.py`.
   - Whole model: U-Net with hengck23's CoordConv decoder.
   - Series model: a shared encoder runs on all 4 rows. Cross-series fusion (conv2d / shared_conv2d / conv3d) is applied
     at encoder levels 1-4.
   - Why fusion helps: leads are related (Einthoven: I + III = II) and share timing, so a blotted peak in one row can be
     inferred from the others.
4. **Loss and training loop.** Read `train.py` and `src/training/trainer.py`.
   - BCE with `pos_weight=20`, because only ~2 px per column are positive.
   - AdamW with lr 1e-3 and weight decay 0.01, cosine schedule, 50 epochs, batch size 4.
   - Mixed precision and gradient checkpointing. The best checkpoint is chosen by validation SNR.
5. **Pixels to signal, and the metric.** Read `Trainer._pixel_to_series`, `src/evaluation/` and the competition
   evaluation page.
   - Post-processing uses a probability-weighted mean y per column, then `scipy.signal.resample`.
   - The official metric aligns each record by up to 0.2 s and removes vertical offset. It sums power over all 12 leads,
     averages records in linear units, then converts to dB.
   - The validation SNR in `trainer.py` is a simpler proxy: per row, with no alignment.
6. **Inference and ensembling.** Read `reference/takashisomeya-2nd-place-submission/`.
   - 2 Whole + 4 Series models, each with horizontal-flip TTA.
   - Series windows are pasted back into the whole frame with 8 px edges ignored.
   - Result: 23.37 public / 23.27 private LB.
7. **Recreate training on a GPU.** Pick a scale that fits the compute (encoder, folds, epochs).
   Compare with the author's numbers: single models 22.5-23.1 dB LB, ensemble 23.27 dB.

## 5. Issues found during setup

| # | Issue | Status |
|---|---|---|
| 1 | `takashisomeya/physionet-ecg-kaggle-data-csv` (ground truth) is private, so the last step of `download_data.sh` fails with 403 | Use competition `train/{id}/{id}.csv` (script does this) |
| 2 | timm >= 1.0.29 rejects names like `tu-timm/tf_efficientnet_b0.ns_jft_in1k` (configs and the author's checkpoints use them) | Pinned `timm<1.0.29` in `requirements.txt` |
| 3 | `Trainer` always enabled autocast, so CPU runs used bf16. That is ~125x slower than fp32 on this CPU (no native bf16) | Fixed: autocast follows `training.amp`, which defaults to on for CUDA and off for CPU |
| 4 | Training resized rectified images to exactly 5600 px wide, but the masks and the author's inference use `x_scale = 5000/1962` (resize to 5607, crop to 5600). Measured image/mask drift was 0 px at the left edge and 4-7 px at the right | Fixed: `resize_rectified()` in `src/data/transforms.py` matches inference. Drift is now within ±1-2 px. A perfect mask gives 41-80 dB SNR against ground truth (lower for 256/512 Hz records) |
| 5 | `predict.py` doesn't pass `df_synthesis` to the dataloader factories (TypeError) and always autocasts | Fix before using `predict.py` |
| 6 | The write-up lists `A.ToGray(p=0.25)`; `transforms.py` has none (all other augmentations match, see `docs/writeups/2nd_place_augmentation.md`) | Tried as modification M2 (`data.grayscale_p=0.25`) |
| 7 | Without ground-truth CSVs, `metric_for_best=snr` falls back to val loss but keeps "higher is better", so `best.pth` would be the worst epoch | Use `training.metric_for_best=loss` until ground truth is present |
| 8 | `SeriesDataset` builds float64 crops (`np.zeros` default dtype): about 8x the memory of uint8 per batch | Performance only |

## 6. Commands

```powershell
conda activate physionet-ecg

# data sample (+ ground truth once the competition rules are accepted)
python scripts/download_sample.py
python scripts/download_sample.py --originals          # raw photos/scans too, to compare with rectified
python scripts/download_sample.py --types 0001 --out data/smoke_folds.csv   # tiny CSV for smoke tests

# full data on a GPU machine (UCC / Colab): all folds + masks + val ground truth, ~62 GB
python scripts/prepare_data.py                 # or --folds 0 1 2 for less data
python train.py --config configs/whole_model.yaml data.csv_path=data/train_folds_available.csv

# CPU smoke test of the full training loop (~3-4 min with smoke_folds.csv)
python train.py --config configs/whole_model.yaml data.csv_path=data/smoke_folds.csv `
    device=cpu num_workers=0 wandb.enabled=false training.num_epochs=1 training.batch_size=1 `
    training.metric_for_best=loss

# Kaggle CLI
kaggle datasets files takashisomeya/physionet-comp-syn-mask-x5000-500hz
kaggle competitions files physionet-ecg-image-digitization
kaggle kernels pull hengck23/demo-submission -p reference/hengck23-demo-submission -m
```
