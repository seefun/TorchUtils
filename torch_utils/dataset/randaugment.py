""" 
RandAugment
Paper: https://arxiv.org/abs/1909.13719
Re-implement (changed) using albumentations by seefun
"""

import numpy as np
import albumentations  # albumentations >= 1.0.0
from torch_utils.dataset.customized_aug import RandomBrightnessNoise, RandomBrightnessContrastPointwise


def randAugment(N=2, M=4, p=1.0, mode="all", cut_out=False):
    """
    Examples:
        >>> # M from 0 to 20
        >>> transforms = randAugment(N=3, M=8, p=0.8, mode='all', cut_out=False)
    """
    # Magnitude(M) search space
    scale = np.linspace(0, 0.4, 20)
    translate = np.linspace(0, 0.4, 20)
    rot = np.linspace(0, 30, 20)
    shear_x = np.linspace(0, 20, 20)
    shear_y = np.linspace(0, 20, 20)
    contrast = np.linspace(0.0, 0.4, 20)
    bright = np.linspace(0.0, 0.4, 20)
    sat = np.linspace(0.0, 0.2, 20)
    hue = np.linspace(0.0, 0.2, 20)
    shar = np.linspace(0.0, 0.9, 20)
    blur = np.linspace(0, 0.2, 20)
    noise = np.linspace(0, 1, 20)
    cut = np.linspace(0, 0.6, 20)
    # Transformation search space
    Aug = [  # geometrical
        albumentations.Affine(scale=(1.0 - scale[M], 1.0 + scale[M]), p=p),
        albumentations.Affine(translate_percent=(-translate[M], translate[M]), p=p),
        albumentations.Affine(rotate=(-rot[M], rot[M]), p=p),
        albumentations.Affine(shear={'x': (-shear_x[M], shear_x[M])}, p=p),
        albumentations.Affine(shear={'y': (-shear_y[M], shear_y[M])}, p=p),
        # Color Based
        albumentations.RandomBrightnessContrast(contrast_limit=contrast[M], p=p),
        albumentations.RandomBrightnessContrast(brightness_limit=bright[M], p=p),
        albumentations.ColorJitter(brightness=0.0, contrast=0.0, saturation=sat[M], hue=0.0, p=p),
        albumentations.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=hue[M], p=p),
        albumentations.Sharpen(alpha=(0.1, shar[M]), lightness=(0.5, 1.0), p=p),
        albumentations.core.composition.PerChannel(
            albumentations.OneOf([
                albumentations.MotionBlur(p=0.5),
                albumentations.MedianBlur(blur_limit=3, p=1),
                albumentations.Blur(blur_limit=3, p=1), ]), p=blur[M] * p),
        albumentations.GaussNoise(var_limit=(0.0 * noise[M], 64.0 * noise[M]), per_channel=True, p=p)
    ]
    # Sampling from the Transformation search space
    if mode == "geo":
        transforms = albumentations.SomeOf(Aug[0:5], N)
    elif mode == "color":
        transforms = albumentations.SomeOf(Aug[5:], N)
    else:
        transforms = albumentations.SomeOf(Aug, N)

    if cut_out:
        cut_trans = albumentations.OneOf([
            albumentations.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=1),
            albumentations.GridDropout(ratio=cut[M], p=1),
            albumentations.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=1),
        ], p=cut[M])
        transforms = albumentations.Compose([transforms, cut_trans])

    return transforms


def segRandAugment(N=3, M1=2, M2=0.25, p=0.5, mode="all", ignore_label=255):
    must_geometrical_aug = []
    other_geometrical_aug = []
    if M1 >= 2:
        must_geometrical_aug.append(albumentations.HorizontalFlip())
    if M1 >= 4:
        must_geometrical_aug.append(albumentations.VerticalFlip())
    if M1 == 1 or M1 == 2:
        other_geometrical_aug.append(albumentations.Affine(
            rotate=(-5, 5), p=p, cval_mask=ignore_label))
        other_geometrical_aug.append(albumentations.Affine(
            scale=(0.9, 1.111), p=p, cval_mask=ignore_label))
        other_geometrical_aug.append(albumentations.Affine(
            translate_percent=(-0.05, 0.05), p=p, cval_mask=ignore_label))
        other_geometrical_aug.append(albumentations.Perspective(
            scale=(0.0, 0.04), p=p, mask_pad_val=ignore_label))
    if M1 >= 3:
        other_geometrical_aug.append(albumentations.Affine(
            scale=(0.8, 1.25), p=p, cval_mask=ignore_label))
        other_geometrical_aug.append(albumentations.Affine(
            translate_percent=(-0.1, 0.1), p=p, cval_mask=ignore_label))
        other_geometrical_aug.append(albumentations.Perspective(
            scale=(0.0, 0.08), p=p, mask_pad_val=ignore_label))
        other_geometrical_aug.append(albumentations.Affine(
            shear={'x': (-5, 5), 'y': (-5, 5)}, p=p, cval_mask=ignore_label))
        if M1 == 3:
            other_geometrical_aug.append(albumentations.Affine(
                rotate=(-30, 30), p=p, cval_mask=ignore_label))
        else:
            other_geometrical_aug.append(albumentations.Affine(
                rotate=(-180, 180), p=p, cval_mask=ignore_label))
    if M1 == 5:
        other_geometrical_aug.append(albumentations.OpticalDistortion(
            distort_limit=0.2, p=p, border_mode=0, value=0, mask_value=ignore_label))

    color_aug = [
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(brightness_limit=M2 * 0.4, p=1),
            albumentations.RandomGamma(gamma_limit=(
                100 - int(40 * M2), 100 + int(40 * M2)), p=1),
            RandomBrightnessNoise(noise_strength=int(3 * M2), max_delta=0.4 * M2)
        ], p=p),
        albumentations.RandomBrightnessContrast(contrast_limit=M2 * 0.4, p=p),
        albumentations.ColorJitter(
            brightness=0.0, contrast=0.0, saturation=M2 * 0.2, hue=0.0, p=p),
        albumentations.ColorJitter(
            brightness=0.0, contrast=0.0, saturation=0.0, hue=M2 * 0.05, p=p),
        albumentations.ImageCompression(
            quality_lower=int(100 - M2 * 39), quality_upper=99, p=p),
        albumentations.OneOf([
            RandomBrightnessContrastPointwise(
                brightness=M2 * 0.1, contrast=M2 * 0.1, p=1),
            albumentations.GaussNoise((0, 64 * M2), p=1)
        ], p=p)
    ]

    # Sampling from the Transformation search space
    if mode == "geo":
        transforms = albumentations.SomeOf(other_geometrical_aug, N)
    elif mode == "color":
        transforms = albumentations.SomeOf(color_aug, N)
    else:
        transforms = albumentations.SomeOf(
            other_geometrical_aug + color_aug, N)

    flip = albumentations.Compose(must_geometrical_aug)
    transforms = albumentations.Compose([flip, transforms])

    return transforms
