def param_groups_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    no_weight_decay_anno = []
    if hasattr(model, 'no_weight_decay'):
        no_weight_decay_anno += model.no_weight_decay()
    for m in model.named_modules():
        if hasattr(m[1], 'no_weight_decay'):
            sub_no_wd_anno = m[1].no_weight_decay()
            no_weight_decay_anno += list(map(lambda x: m[0] + '.' + x, sub_no_wd_anno))
    no_weight_decay_anno = set(no_weight_decay_anno)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_anno:
            no_decay.append(param)
        elif 'cls_token' in name or 'pos_embed' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


"""
optimizer = torch.optim.AdamW(
            params=tu.param_groups_weight_decay(model, weight_decay=1e-5),
            lr=3e-4)
"""
