import albumentations
from .randaugment import randAugment
from albumentations import pytorch as AT

IMAGE_SIZE = 512

train_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    # albumentations.RandomRotate90(p=0.5),
    # albumentations.Transpose(p=0.5),
    # albumentations.Flip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    randAugment(image_size=IMAGE_SIZE, N=2, M=12, p=0.9, mode='all', cut_out=False),
    albumentations.Normalize(),
    albumentations.Cutout(num_holes=8, max_h_size=IMAGE_SIZE//8, max_w_size=IMAGE_SIZE//8, fill_value=0, p=0.25),
    AT.ToTensorV2(),
    ])

test_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.Normalize(),
    AT.ToTensorV2(),
    ])
