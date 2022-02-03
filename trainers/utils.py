import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
class ContrastiveCrossEntropyLoss(torch.nn.Module):
    def __init__(self, contastive_num=2):
        super(ContrastiveCrossEntropyLoss, self).__init__()
        self.contastive_num = contastive_num
    def forward(self, logits, target):
        target = target.view(-1, 1)
        # print("target:",target.shape)
        logits = F.log_softmax(logits, 1)  #target类的概率
        # print("logits:",logits.shape)

        max_ids = logits.argsort(dim=1,descending=True)[:,:self.contastive_num] #对应的非target类

        loss =  -1 * logits.gather(1, target)    + logits.gather(1, max_ids)
        loss = loss.sum()
        return loss