
# TorchUtils 

TorchUtils is a pytorch lib with several useful tools and training tricks. (Work In Progress)

## Install
```
pip install -r requirements.txt
pip install .
```

## Import

```
import torch_utils as tu
```


## Seed All

```
SEED = 42
tu.tools.seed_everything(SEED)
```

## Data Augmentation

```
train_transform = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.HorizontalFlip(p=0.5),
    tu.dataset.randAugment(image_size=IMAGE_SIZE, N=2, M=12, p=0.9, mode='all', cut_out=False),
    albumentations.Normalize(),
    albumentations.Cutout(num_holes=8, max_h_size=IMAGE_SIZE//8, max_w_size=IMAGE_SIZE//8, fill_value=0, p=0.25),
    AT.ToTensorV2(),
    ])

mixup_dataset = tu.dataset.MixupDataset(dataset, alpha=1.0, prob=0.1, mixup_to_cutmix=0.3) 
# 0.07 mixup and 0.03 cutmix
```

## Model

recommanded pretrained models:

- [ResNeSt](https://github.com/zhanghang1989/ResNeSt)  
- SEResNext-50 
- GPU-Efficient
- swsl_ResNeXt
- [BiT/ResNetV2](https://github.com/google-research/big_transfer) 
- [TResNet](https://github.com/mrT23/TResNet)
- EfficientNet_ns
- [ResNext_WSL](https://github.com/facebookresearch/WSL-Images)
- MixNet
- SKNet
- [SGENet](https://github.com/implus/PytorchInsight)
- [HRNet](https://github.com/HRNet)
- Res2Net


recommanded github repos：

- [pytorch-image-models(timm)](https://github.com/rwightman/pytorch-image-models)
- [imgclsmob(pytorchcv)](https://github.com/osmr/imgclsmob/tree/master/pytorch)
- [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)
- [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [pytorch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [pretrained-models-pytorch](https://github.com/Cadene/pretrained-models.pytorch)



fast build models with torch_utils: 
```
model = tu.ImageModel(name='resnest50d', pretrained=True, 
                      pooling='concat', fc='multi-dropout', 
                      feature=2048, classes=1))
model.cuda()
```


```
import timm

model = timm.create_model('tresnet_m', pretrained=True)
model.global_pool = tu.layers.FastGlobalConcatPool2d(flatten=True)
model.head = tu.layers.get_attention_fc(2048*2, 1) 
model.cuda()
```

```
from pytorchcv.model_provider import get_model as ptcv_get_model

model = ptcv_get_model('seresnext50_32x4d', pretrained=True)
model.features.final_pool = tu.layers.GeM() 
model.output = tu.layers.get_simple_fc(2048, 1)   
model.cuda()
```

model utils:
```
# model summary
tu.summary(model, input_size=(batch_size, 1, 28, 28))

# 3 channels pretrained weights to 1 channel
weight_rgb = model.conv1.weight.data
weight_grey = weight_rgb.sum(dim=1, keepdim=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=xxx, stride=xxx, padding=xxx, bias=False)
model.conv1.weight.data = weight_grey

# 3 channels pretrained weights to 4 channel
weight_rgb = model.conv1.weight.data
weight_y = weight_rgb.mean(dim=1, keepdim=True)
weight_rgby = torch.cat([weight_rgb,weight_y], axis=1) * 3 / 4
model.conv1 = nn.Conv2d(4, 64, kernel_size=xxx, stride=xxx, padding=xxx, bias=False)
model.conv1.weight.data = weight_rgby

# 2D models to 3d models using ACSConv (advanced)
## using code in this repo: https://github.com/M3DV/ACSConv
```


## Optimizer
```
optimizer_ranger = tu.Ranger(model_conv.parameters(), lr=LR)

# optimizer = torch.optim.AdamW(model_conv.parameters(), lr=LR, weight_decay=2e-4)
```


## Criterion
```
# for example:
criterion = tu.LabelSmoothingCrossEntropy()
```


## Find LR 
```
lr_finder = tu.LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=500, accumulation_steps=1)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```


## LR Scheduler
```
scheduler = tu.CosineAnnealingWarmUpRestarts(optimizer, T_0=T, T_mult=1, eta_max=LR, T_up=0, gamma=0.05)

# torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1)

# torch.optim.lr_scheduler.OneCycleLR or tu.OneCycleScheduler
```


## AMP

Ref: https://pytorch.org/docs/master/notes/amp_examples.html


## TODO
- [ ] [duplicate image detection](https://www.kaggle.com/nakajima/duplicate-train-images?scriptVersionId=47295222) 
- [ ] [Ranger21](https://github.com/lessw2020/Ranger21) optimizer and lr_scheduler
- [ ] KD: [torchdistill](https://github.com/yoshitomo-matsubara/torchdistill)

