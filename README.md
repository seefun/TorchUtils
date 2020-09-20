
# TorchUtils 

TorchUtils is a pytorch lib with several useful tools and some state-of-the-art training methods or tricks. (Work In Progress)


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

TODO:

- [x] common data augmentations used in competition
- [ ] [Automold--Road-Augmentation-Library](https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library)
- [ ] [GridMask](https://www.kaggle.com/haqishen/gridmask)
- [ ] [AugMix](https://www.kaggle.com/haqishen/augmix-based-on-albumentations)


## Model

recommanded pretrained models:

- SEResNext-50 
- [TResNet](https://github.com/mrT23/TResNet)
- [ResNeSt](https://github.com/zhanghang1989/ResNeSt)
- EfficientNet 
- [ResNext_WSL](https://github.com/facebookresearch/WSL-Images)
- [BiT](https://github.com/google-research/big_transfer) 
- MixNet
- SKNet
- [SGENet](https://github.com/implus/PytorchInsight)
- [HRNet](https://github.com/HRNet)

from github repos：

- [pytorch-image-models(timm)](https://github.com/rwightman/pytorch-image-models)
- [imgclsmob(pytorchcv)](https://github.com/osmr/imgclsmob/tree/master/pytorch)
- [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)
- [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [pytorch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [pretrained-models-pytorch](https://github.com/Cadene/pretrained-models.pytorch)



fast build models with torch_utils: 

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
tu.models.summary(model, (3,224,224))

# 3 channels pretrained weights to 1 channel
weight_rgb = model.conv1.weight
weight_grey = weight_rgb.sum(dim=1, keepdim=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=xxx, stride=xxx, padding=xxx, bias=False)
model.conv1.weight = torch.nn.Parameter(weight_grey)

# 2D models to 3d models using ACSConv (advanced)
## using code in this repo: https://github.com/M3DV/ACSConv
```


## Optimizer
```
optimizer_ranger = tu.Ranger(model_conv.parameters(), lr=LR)

# optimizer = torch.optim.AdamW(model_conv.parameters(), lr=LR, weight_decay=2e-4)
```

## Criterion
TODO:
- [ ] Criterions



## Find LR 
```
lr_finder = tu.LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```


## LR Scheduler
```
scheduler = tu.CosineAnnealingWarmUpRestarts(optimizer, T_0=T, T_mult=1, eta_max=LR, T_up=0, gamma=0.05)

# torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1)

# torch.optim.lr_scheduler.OneCycleLR
# tu.OneCycleScheduler
```



## TTA:
TODO ：
- [ ] TTA: https://github.com/qubvel/ttach

## AMP
TODO:
In pytorch 1.6
https://pytorch.org/docs/master/notes/amp_examples.html


## TODO
1. clean code using pytorch 1.6.0
2. cutmix : https://github.com/ildoonet/cutmix
3. randaug: https://github.com/ildoonet/pytorch-randaugment
4. fast-autoaug: https://github.com/kakaobrain/fast-autoaugment
5. SupContrast: https://github.com/HobbitLong/SupContrast
6. metric learning: https://github.com/KevinMusgrave/pytorch-metric-learning