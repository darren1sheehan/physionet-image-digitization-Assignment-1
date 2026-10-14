# Other top solutions: sources and ideas

Collected 14 Oct 2026 for the lab report ("other winning solutions in the context of this one") and to choose
modifications. The solution replicated is the **2nd place** (Takashi Someya).

| Place | Source | Core approach |
|---|---|---|
| 1st | [1st_place.md](1st_place.md) | hengck23 rectification (improved, high-res remap) -> **direct regression** of signal values (SmoothL1), 10-model ensemble |
| 2nd | Someya write-up, code in this repo | hengck23 rectification -> **segmentation** (probability map, BCE) -> weighted mean y per column; Whole + Series models |
| 3rd | [3rd_place.md](3rd_place.md) | hengck23 rectification at high res -> segmentation (ConvNeXt V2 / HRNet, BCE) -> **parabolic sub-pixel** decoding |
| 4th | [GitHub uchiyama33/physionet_4th_place](https://github.com/uchiyama33/physionet_4th_place) | own grid intersection + line-index models -> rectified images -> digitization; FCMAE pre-training (ConvNeXt V2), external CODE-15 / PTB-XL data |
| 5th | [GitHub dangnh0611/kaggle_ecg_digitization](https://github.com/dangnh0611/kaggle_ecg_digitization), [ablations](https://github.com/dangnh0611/kaggle_ecg_digitization/blob/main/docs/ABLATION_STUDY.md) | orientation -> keypoints (pseudo-labels) -> per-lead crops -> **column-wise Gaussian heatmaps** with column-wise JSD/CE loss |
| 7th | [Kaggle write-up](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/7th-place-solution) (not copied; summary from search) | rotation, lead detection/segmentation, ViT-small digitizer; **only horizontal-flip augmentation** |
| 11th | [11th_place.md](11th_place.md) | hengck23 + iterative grid alignment -> multi-scale U-Nets, **SNR-aware loss** (+1.6 dB) |

## Shared innovations

- hengck23's Stage 0/1 rectification: 1st, 2nd, 3rd, 11th.
- Horizontally stretched rectified pages for finer time sampling: 2nd (5600 px), 3rd (4400 px), 11th (6600 px option).
- Fourier resampling (`scipy.signal.resample`): 1st, 2nd.
- Coordinate information in the network: 1st (coordinate channels), 2nd (CoordConv decoder).

## Conflicting evidence (worth discussing)

| Idea | Helped | Did not help |
|---|---|---|
| Lead II fusion + Einthoven / Goldberger correction | 1st (+0.06 dB, with TTA +0.07), 3rd (used, weight II = 2) | 11th (SNR decreased) |
| Gaussian-blurred (soft) targets | 5th (sigma 1.5 px best) | 3rd |
| Heavy augmentation | 1st, 3rd | 7th (horizontal flip only) |
| BCE loss | 2nd, 3rd (used) | 5th (column-wise CE/JSD better by about 0.2 dB) |
