def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        
# usage: model.apply(set_bn_eval) # this will freeze the bn in training process
