import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class MyContrastiveLoss(nn.Module):
    def __init__(self, device, margin=0.2, max_violation=False):
        super().__init__()
        self.device = device
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # input img:batch_size,36,d
        # input txt : batch_size,n,d

        # -----------------------构建三元组损失，返回loss值----------------------------------
        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        # clear diagonals
        mask = torch.eye(scores.size(0)).to(self.device) > .5
        # I = mask
        I = Variable(mask)

        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        # keep the maximum violating negative for each query,
        # Use max instead of sum in the rank loss.
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')