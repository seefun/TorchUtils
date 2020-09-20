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

# transforms_train = A.Compose([
#     A.Transpose(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightness(limit=0.2, p=0.75),
#     A.RandomContrast(limit=0.2, p=0.75),
#     A.OneOf([
#         A.MotionBlur(blur_limit=5),
#         A.MedianBlur(blur_limit=5),
#         A.GaussianBlur(blur_limit=5),
#         A.GaussNoise(var_limit=(5.0, 30.0)),
#     ], p=0.7),

#     A.OneOf([
#         A.OpticalDistortion(distort_limit=1.0),
#         A.GridDistortion(num_steps=5, distort_limit=1.),
#         A.ElasticTransform(alpha=3),
#     ], p=0.7),

#     A.CLAHE(clip_limit=4.0, p=0.7),
#     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
#     A.Resize(image_size, image_size),
#     A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),    
#     A.Normalize()
# ])

# transforms_val = A.Compose([
#     A.Resize(image_size, image_size),
#     A.Normalize()
# ])
