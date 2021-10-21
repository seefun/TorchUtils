def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def freeze_bn(model):
    for m in model.named_modules():
        set_bn_eval(m[1])


def unfreeze_bn(model):
    for m in model.named_modules():
        set_bn_train(m[1])

# usage: model.apply(freeze_bn) # this will freeze the bn in training process
