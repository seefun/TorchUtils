""" 
RandAugment
Paper: https://arxiv.org/abs/1909.13719
Re-implement using albumentations by seefun
"""

import numpy as np
import albumentations

## TODO change augmentation settings (adapt to image size)

def randAugment(image_size, N, M, p, mode="all", cut_out = False):
    """
    Examples:
        >>> # M from 0 to 20
        >>> transforms, ops = randAugment(image_size=512, N=3, M=12, p=0.9, mode='all', cut_out=False)
    """
    # Magnitude(M) search space  
    shift_x = np.linspace(0,image_size*0.5,20)
    shift_y = np.linspace(0,image_size*0.5,20)
    rot = np.linspace(0,30,20)
    shear = np.linspace(0,20,20)
    sola = np.linspace(0,160,20)
    post = [0,0,1,1,2,2,3,3,4,4,4,4,5,5,6,6,7,7,8,8]
    contrast = np.linspace(0.0,0.4,20)
    bright = np.linspace(0.0,0.4,20)
    shar = np.linspace(0.0,0.9,20)
    blur = np.linspace(0,0.5,20)
    cut = np.linspace(0,0.6,20)
     # Transformation search space
    Aug =[# geometrical
        albumentations.ShiftScaleRotate(shift_limit_x=shift_x[M], rotate_limit=0,   shift_limit_y=0, shift_limit=shift_x[M], p=p),
        albumentations.ShiftScaleRotate(shift_limit_y=shift_y[M], rotate_limit=0, shift_limit_x=0, shift_limit=shift_y[M], p=p),
        albumentations.IAAAffine(rotate=rot[M], p=p),
        albumentations.IAAAffine(shear=shear[M], p=p),
        # Color Based
        albumentations.Solarize(threshold=sola[M], p=p),
        albumentations.Posterize(num_bits=post[M], p=p),
        albumentations.RandomContrast(limit=contrast[M], p=p),
        albumentations.RandomBrightness(limit=bright[M], p=p),
        albumentations.IAASharpen(alpha=(0.1, shar[M]), lightness=(0.5, 1.0)),
        albumentations.core.composition.PerChannel(
            albumentations.OneOf([
                albumentations.MotionBlur(p=1),
                albumentations.MedianBlur(blur_limit=3, p=1),
                albumentations.Blur(blur_limit=3, p=1),])
            , p=blur[M]*p),]
    # Sampling from the Transformation search space
    if mode == "geo": 
        ops = np.random.choice(Aug[0:5], N)
    elif mode == "color": 
        ops = np.random.choice(Aug[5:], N)
    else:
        ops = np.random.choice(Aug, N)
  
    if cut_out:
        cut_trans = albumentations.OneOf([
            albumentations.CoarseDropout(max_holes=8, max_height=image_size//16, max_width=image_size//16, fill_value=0, p=1),
            albumentations.GridDropout(ratio=cut, p=1),
            albumentations.Cutout(num_holes=8, max_h_size=image_size//16, max_w_size=image_size//16, p=1),
        ], p=1), 
        ops.append(cut_trans, p=p*cut)
    transforms = albumentations.Compose(ops)
    return transforms, ops