import albumentations
from albumentations import pytorch as AT

train_transform = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.Flip(p=0.5),
    albumentations.OneOf([
        albumentations.RandomBrightness(0.2, p=1), 
        albumentations.RandomContrast(0.2, p=1),
        albumentations.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1),
    ], p=0.5), 
    albumentations.OneOf([
        albumentations.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3)),
        albumentations.IAASharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0)),
        albumentations.GaussNoise((5,25)),
    ], p=0.5), 
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=15, border_mode=1, p=0.5),
    albumentations.Cutout(num_holes=4, max_h_size=36, max_w_size=36, fill_value=255, p=0.25),
    
    albumentations.Normalize(),
    AT.ToTensorV2(),
    ])

test_transform = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(),
    AT.ToTensorV2(),
    ])