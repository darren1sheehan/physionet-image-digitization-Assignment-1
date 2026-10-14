# 11th place solution: multi-scale approach and grid point alignment

Source: zx2 (xun1993), fengxing, zacarioli. *11th Place Solution*. Kaggle write-up, 28 Jan 2026.
https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/11th-place-solution
(Copied from the page on 14 Oct 2026 for reference.)

---

## Overview

The pipeline aims to improve SNR by combining lead reduction, stronger network architectures, SNR-aware loss
functions and post-processing grid alignment. Stage 1 uses hengck23's notebook, with the generated grid points
refined iteratively based on equal spacing in x and y. Most effort went into the second stage.

## Lead wire cropping

Four long lead strips are cropped from the standard 12-lead page. This reduces computation, speeds up validation and
allows larger networks. The full lead II serves as the reference for evaluating strategies.

## SNR-aware loss

An SNR-based loss term is added to the training objective. It optimises signal quality directly and contributed an
SNR improvement of **about 1.6 dB**.

## Backbone

ResNet34 replaced by ConvNeXt Large: **+0.6 dB**.

## Iterative grid point alignment

Detected grid points are refined with an iterative spacing correction. Most effective for image types 3, 4, 11 and 12
(small local distortion); type 6 improves slightly. Average **+0.35 dB** for the supported types.

## Multi-scale and multi-model ensemble

Three input sizes in training and inference: (2200, 1700), (6600, 1700), (4400, 3400). Architectures: ConvNeXt
Large, EfficientNet, U-Net.

## Effective

- Larger backbone (ConvNeXt Large)
- SNR-aware loss
- Iterative grid alignment (types 3, 4, 6, 11, 12)
- Multi-scale training and model ensemble

## Ineffective

- Extensive data generation, CycleGAN-based style transfer
- **Correcting the output with I + III = II and aVF + aVR + aVL = 0 decreased SNR.**
