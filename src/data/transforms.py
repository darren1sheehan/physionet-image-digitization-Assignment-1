"""Data augmentation transforms using albumentations."""
from typing import Optional

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Rectified pages are 1700 x 2200 and 10 s of ECG spans x=118..2080. Stretch so that
# span is exactly 5000 px (= 500 Hz samples), matching the masks and the author's inference.
X_SCALE = 5000 / (2080 - 118)
RESIZED_W = int(2200 * X_SCALE) + 1  # 5607, cropped to 5600 by the datasets


def resize_rectified(image):
    """Resize a rectified RGB page to the mask frame (1700, RESIZED_W)."""
    return cv2.resize(image, (RESIZED_W, 1700), interpolation=cv2.INTER_LINEAR)


def get_train_transforms(
    height: int = 480,
    width: int = 5600,
) -> A.Compose:
    """Get training data augmentation transforms.

    Args:
        height: Target height for resize.
        width: Target width for resize.

    Returns:
        Albumentations Compose object.
    """
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.2,
        ),
        A.RandomShadow(p=0.2),
        A.GaussianBlur(p=0.2),
        A.CoarseDropout(
            num_holes_range=(1,8),
            hole_height_range=(0.01, 0.1),
            hole_width_range=(0.01, 0.05),
            fill_mask=None,
            p=0.1
        ),
    ]

    return A.Compose(transforms)


def get_val_transforms(
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> A.Compose:
    """Get validation data transforms (no augmentation).

    Args:
        height: Target height for resize (optional).
        width: Target width for resize (optional).

    Returns:
        Albumentations Compose object.
    """
    transforms = []

    if height is not None and width is not None:
        transforms.append(A.Resize(height, width))

    return A.Compose(transforms)


def apply_transform(image, mask, transform):
    """Apply albumentations transform to image and mask.

    Args:
        image: Image array (H, W, C) or (H, W).
        mask: Mask array (H, W) or (H, W, C).
        transform: Albumentations transform.

    Returns:
        Transformed (image, mask) tuple.
    """
    if transform is None:
        return image, mask

    # Handle multi-channel masks
    if mask.ndim == 3 and mask.shape[0] < mask.shape[-1]:
        # (C, H, W) -> (H, W, C)
        mask = mask.transpose(1, 2, 0)
        transpose_back = True
    else:
        transpose_back = False

    result = transform(image=image, mask=mask)
    transformed_image = result["image"]
    transformed_mask = result["mask"]

    if transpose_back:
        transformed_mask = transformed_mask.transpose(2, 0, 1)

    return transformed_image, transformed_mask
