# 2nd place (Takashi Someya): augmentation, as stated in the write-up

Source: Someya, T. *2nd Place Solution*, Kaggle write-up, "Augmentation" section.
https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution
(Copied 14 Oct 2026.)

```python
image_only_aug = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.2),
    A.RandomShadow(p=0.2),
    A.GaussianBlur(p=0.2),
    A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(0.01, 0.1), hole_width_range=(0.01, 0.05), fill=0, p=0.1),
    A.ToGray(p=0.25),
])
image_and_mask_aug = A.HorizontalFlip(p=0.5)
```

## Compared with the public code (`src/data/transforms.py`, `get_train_transforms`)

| Augmentation | Write-up | Code |
|---|---|---|
| HorizontalFlip (image and mask) | p=0.5 | p=0.5 |
| RandomBrightnessContrast | +-0.1, p=0.2 | +-0.1, p=0.2 |
| RandomShadow | p=0.2 | p=0.2 |
| GaussianBlur | p=0.2 | p=0.2 |
| CoarseDropout | 1-8 holes, p=0.1, image only | same, `fill_mask=None` (mask untouched) |
| **ToGray** | **p=0.25** | **missing** |

ToGray is the only difference. Modification M2 restores it with `data.grayscale_p=0.25`.
