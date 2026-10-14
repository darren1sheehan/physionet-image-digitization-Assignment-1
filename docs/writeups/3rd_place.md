# 3rd place solution: high-resolution mapping + parabolic refinement

Source: Tang, hirotetsu. *3rd place solution*. Kaggle write-up, 23 Jan 2026.
https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/3rd-place-solution
Training code: https://github.com/tanghaozhe/physionet-ecg-image-digitization-3rd-place
Submission notebook: https://www.kaggle.com/code/hirotetsu/physionet-submission-3rd-place
(Copied from the page on 14 Oct 2026 for reference; the figures are not included.)

---

Thanks to PhysioNet and Kaggle for organizing this competition. Special thanks to @hengck23 for sharing the image
rectification pipeline.

## Overview

- **High-resolution mapping rectification**: calculate geometric parameters using low-resolution images, then scale
  them up and apply them directly to the original high-resolution images, to avoid losing detail.
- **Training**: strong data augmentation to make the model more robust.
- **Sub-pixel precision**: a parabolic refinement turns pixel-level predictions into floating-point values.

## Mask preparation

A one-pixel-wide line is drawn from the signal data. Making the line wider with a Gaussian blur (softer labels) did
**not** improve results, so they kept the one-pixel mask and intentionally trained an "overfitted" model that
predicts thin waveforms with high confidence. The same mask is used for all images of the same record.

| Image size | SNR (dB) of the mask itself | Notes |
|---|---|---|
| 2200 x 1700 | 24.2235 | standard dimension (200 DPI) |
| 4400 x 1700 | 31.2271 | selected |

- **Wider images**: doubling the width greatly improves SNR. ECG signals are time series, so more horizontal pixels
  mean better sampling and less quantization error.
- **Fixed height**: keeping the waveform about 1 px thick helps the model locate the signal and stabilises
  post-processing.

Larger masks usually give better SNR, but inference memory limited them to doubling the width. To keep image quality,
the image transformation was separated from the rectification.

## High-resolution mapping rectification

The rectification model outputs relatively small rectified images, which loses quality. The process is split into
(1) parameter estimation at low resolution and (2) image transformation at high resolution: the homography matrix
and grid coordinates are scaled to the original raw image size. This gave a significant improvement.

## Signal segmentation model

- **Encoders**: ConvNeXt V2 > HRNet > EfficientNet V2. The final submission ensembles ConvNeXt V2 and HRNet by a
  weighted average of the **post-processed signals**, not of the segmentation probabilities; this signal-level
  ensembling was better.
- **Decoder**: UnetDecoder.
- **Augmentation** (very aggressive, since some hidden test samples may be very noisy): GridDropout, CoarseDropout,
  AddPerlinDirt, MotionBlur, Downscale, GaussianBlur, ImageCompression, GaussNoise, ISONoise, ToGray,
  RandomBrightnessContrast, HueSaturationValue, RandomShadow, ElasticTransform (small, to simulate distortions from
  unwarping), HorizontalFlip, VerticalFlip.
- **Loss**: BCE. Even at the lowest loss the predicted waveforms were too thick, so training was deliberately
  continued into an "overfitted" model, selected by Dice score and CV score.

## Sub-pixel signal processing

- **Weighted centroid**: take the top-K pixels and compute the position weighted by probability. Good, but sensitive
  to K and the weighting exponent.
- **Parabolic interpolation** (chosen): refine the integer argmax by fitting a parabola to the peak and its two
  neighbours. Consistently the best, because the model outputs very thin, sharp lines.

    delta = (y_left - y_right) / (2 * (y_left - 2*y_center + y_right))
    position = integer index + delta

## Others

- **Lead II fusion**: merge the short lead II prediction with the long rhythm strip (series 3) by averaging the
  overlapping head region.
- **Einthoven correction**: adjust leads I, II and III to enforce II = I + III, with lead II weighted 2 and leads I and
  III weighted 1.
- **TTA**: original, horizontal flip, vertical flip, and both flips.
