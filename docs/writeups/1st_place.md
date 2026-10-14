# 1st place solution: direct regression without signal mask

Source: outrunner, Yijun, Arya, Bob, Shane Hsieh. *1st place solution*. Kaggle write-up, 23 Jan 2026.
https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/1st-place-solution
(Copied from the page on 14 Oct 2026 for reference; the figures are not included.)

---

We would like to thank PhysioNet and Kaggle for organizing this ECG signal prediction competition. Special thanks
to @hengck23 for sharing the image rectification pipeline. This pipeline includes Stage 0 (image rotation and
homography) and Stage 1 (rectifying homography-transformed images). We have integrated both stages into our
workflow and developed a strategy to predict all ECG leads directly from these rectified images.

## Overview

**Prediction based on rectified images.** Transforming various image styles into a standardized format minimizes
inconsistencies from different perspectives. This helps the model ignore spatial distortions and focus on essential
heart patterns, resulting in more accurate and reliable ECG predictions.

**Remapping from high resolution image.** Input images are generated using two different approaches to maximize
feature preservation:
1. Homography-based: rectifying the image at a high resolution before remapping.
2. Direct remapping: skips the homography conversion and performs remapping directly from the source.

**Resampling in the Fourier domain.** Instead of linear interpolation, resampling is performed in the Fourier domain
using `scipy.signal.resample`. This method is better suited for ECG signals.

**Data preparation.** To validate the model, the first ten training samples were used as a validation set.

## Pipeline

1. **Preprocessing.** Two methods were used to generate input images for the prediction model. The first transforms
   the rotated image into a 3200x2400 resolution via a homography matrix, followed by grid point remapping. The
   second rescales the grid points using the homography matrix and performs remapping directly on the rotated
   image. In both cases, the lower region containing the ECG signals is cropped for use while the top portion
   containing personal information is discarded. Images are then converted to grayscale and concatenated with
   coordinate-based features as model inputs, providing the network with both visual intensity and spatial context.
2. **Prediction strategy 1.** Features from the last and second-to-last blocks of the backbone are used as inputs
   for numerical prediction. Since the ECG images consist of four vertically stacked data groups, these features
   are partitioned by height into four segments. These segments are then expanded to four times the input width.
3. **Prediction strategy 2.** Features from the last block of the backbone, partitioned by height into four
   segments and expanded to four times the input width.
4. **Prediction strategy 3.** Features from the last block; instead of partitioning by height, the height dimension
   is merged with max pooling, the channels are split into four segments and expanded to four times the input width.
5. **Post-processing.** The middle 10,000, 15,000 or 20,000 outputs are selected and resampled. Three types of lead
   blending are then performed, based on ECG characteristics and Einthoven's law:
   - mix the first quarter of the long II with the II;
   - apply `II = I + III` if `np.abs(II-I-III).mean() < 0.01`;
   - apply `aVR + aVL + aVF = 0` if `np.abs(aVR+aVL+aVF).mean() < 0.01`.
6. **TTA.** Three test-time augmentations: gamma adjustments (1.0, 0.9 and 1.1) for brightness, plus cropping and
   resizing of the Stage 1 inputs. Cropping: Stage 1 inputs resized to 1560x1248, then a centre crop of 1440x1152.
   Resizing: Stage 1 input resolution resized to 1600x1280.
7. **Training details.** SmoothL1Loss, AdamW, batch size 1, learning rate 1e-4 with a cosine schedule.

**Augmentation.** Synthetic blank ECG images were generated to prevent the model from predicting anomalous values
when encountering occlusions or noise. Random variations were also applied to the ground truth data to increase
diversity. Moiré patterns, flares and overlaid non-rectified images were introduced to increase training difficulty
and robustness.

## Results

Ten models were ensembled. For validation, 30 images were selected by taking three segments (0006, 0009 and 0012)
from ten image ids. **TTA improved overall performance by 0.07, and the Einthoven's-law correction gave a further
0.06.**

| Model | Validation | LB (Private) | Backbone | Input size | Output size | Strategy | Ensemble weight |
|---|---|---|---|---|---|---|---|
| 1 | 26.79 | | convnextv2_tiny | 5056x1280 | 4x20000 | 1 | 20 |
| 2 | 26.21 | | convnextv2_base | 2528x1280 | 4x10000 | 1 | 32 |
| 3 | 26.50 | | tf_efficientnetv2_m | 5056x1280 | 4x20000 | 2 | 30 |
| 4 | 26.03 | | hgnetv2_b6 | 5056x1280 | 4x20000 | 2 | 11 |
| 5 | 26.46 | | mambaout_base | 5056x1280 | 4x20000 | 2 | 8 |
| 6 | 25.65 | | caformer_m36 | 2528x1280 | 4x10000 | 2 | 9 |
| 7 | 26.77 | 22.98 (22.75) | convnextv2_tiny | 5056x1280 | 4x15000 | 3 | 31 |
| 8 | 26.56 | | inception_next_small | 5056x1280 | 4x15000 | 3 | 23 |
| 9 | 26.76 | | inception_next_base | 5056x1280 | 4x20000 | 3 | 27 |
| 10 | 26.78 | | inception_next_base | 5056x1280 | 4x20000 | 1 | 25 |

**Crossing signal.** ECG-Image-Kit was used to select cases where leads overlap or cross. Even where lead V3 and
lead II partially intersect, the model still predicts accurately.

**Resampling** (one image, local): `scipy.signal.resample` 23.834 dB vs linear interpolation 22.176 dB.

**Remapping** (one image, local): from high resolution 23.834 dB vs from low resolution 21.267 dB.

**Rectification.** Two enhancements to the original method: grid point positions computed as a weighted average
over the feature map (floating-point precision), and surface fitting to find and remove outliers, with gaps filled
by interpolation or the regression function.

**Other.** Image type 3 was generated by scanning image type 1, but the offset between grid and signal in type 3
differs from that in type 1.

## Selected comments

- hengck23 (16th): direct regression of the 44x57 rectification from the homography-normalised image is reported
  possible in papers.
- outrunner (1st): they tried regression from the normalised (non-rectified) image, which scored properly but not
  competitively; predicting every pixel's original coordinate and inverse-remapping was good but also not
  competitive enough.
- On the reshape/transpose in the head: it keeps spatial alignment, with output ordered
  `[x0c0, x0c1, ..., x1c0, x1c1, ...]`; the many channels per x let the model encode sub-pixel information.
