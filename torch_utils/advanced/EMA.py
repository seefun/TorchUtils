# SWA in pytorch https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/#how-to-use-swa-in-pytorch

# ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
#         0.1 * averaged_model_parameter + 0.9 * model_parameter
# ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
# for i in range(EPOCH):
#     xxxx
#     if i > ema_start:
#         ema_model.update_parameters(model)
# torch.optim.swa_utils.update_bn(loader, ema_model)
# preds = ema_model(test_input)
