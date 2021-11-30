""" 
RandAugment
Paper: https://arxiv.org/abs/1909.13719
Re-implement (changed) using albumentations by seefun
"""

import numpy as np
import albumentations  # albumentations >= 1.0.0


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
        albumentations.GaussNoise(var_limit=(0.0 * noise[M], 32.0 * noise[M]), per_channel=True, p=p)
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
