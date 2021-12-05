# HRNet with new pretrain
# adaptive hrnet pretrained weight to in_chans:

# def adapt_input_conv(in_chans, conv_weight):
#     conv_type = conv_weight.dtype
#     conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
#     O, I, J, K = conv_weight.shape
#     if in_chans == 1:
#         if I > 3:
#             assert conv_weight.shape[1] % 3 == 0
#             # For models with space2depth stems
#             conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
#             conv_weight = conv_weight.sum(dim=2, keepdim=False)
#         else:
#             conv_weight = conv_weight.sum(dim=1, keepdim=True)
#     elif in_chans != 3:
#         if I != 3:
#             raise NotImplementedError('Weight format not supported by conversion.')
#         else:
#             # NOTE this strategy should be better than random init, but there could be other combinations of
#             # the original RGB input layer weights that'd work better for specific cases.
#             repeat = int(math.ceil(in_chans / 3))
#             conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
#             conv_weight *= (3 / float(in_chans))
#     conv_weight = conv_weight.to(conv_type)
#     return conv_weight

# 1. timm create features only models (in_chans)
# 2. load pretrained models with adaptive input conv
# 3. add fcnhead (hypercolumn->1x1 conv(n->n)->1x1 output(or 3x3 output, could choice, n->out)->align_corner_upsample/pixel shuffle output)
# 4. add ocrhead with aux loss
