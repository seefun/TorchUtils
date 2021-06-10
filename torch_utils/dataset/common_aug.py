import albumentations
from .randaugment import randAugment
from albumentations import pytorch as AT

IMAGE_SIZE = 512

train_transform_randaug = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.RandomRotate90(p=0.5), # albumentations.SafeRotate(border_mode=1, p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    randAugment(),
    albumentations.Normalize(),
    albumentations.Cutout(num_holes=8, max_h_size=IMAGE_SIZE//8, max_w_size=IMAGE_SIZE//8, fill_value=0, p=0.25),
    AT.ToTensorV2(),
    ])   

train_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.RandomBrightness(0.2, p=1), 
        albumentations.RandomContrast(0.2, p=1),
        albumentations.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1),
    ], p=0.5), 
    albumentations.OneOf([
        albumentations.ElasticTransform(alpha=1, sigma=20, alpha_affine=10),
        albumentations.GridDistortion(num_steps=6, distort_limit=0.1),
        albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
    ], p=0.25), 
    albumentations.OneOf([
        albumentations.CLAHE(),
        albumentations.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3)),
        albumentations.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0)),
        albumentations.GaussNoise((5,30)),
        albumentations.JpegCompression(30,90),
    ], p=0.5), 
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=1, p=0.5),
    albumentations.Normalize(),
    albumentations.Cutout(num_holes=8, max_h_size=IMAGE_SIZE//8, max_w_size=IMAGE_SIZE//8, fill_value=0, p=0.25),
    AT.ToTensorV2(),
    ])   

test_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.Normalize(),
    AT.ToTensorV2(),
    ])
