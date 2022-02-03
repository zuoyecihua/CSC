import torch
from torchcontrib.optim import SWA


def get_optimizer(model, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]


    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    if args.use_swa:
        optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=5e-5)

    return optimizer