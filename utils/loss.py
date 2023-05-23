import torch.nn as nn
import torch.nn.functional as F

class loss_cls_kd(nn.Module):
    def __init__(self, params):
        super(loss_cls_kd, self).__init__()
        self.alpha = params.alpha
        self.T = params.temperature
    def forward(self, output, target, output_tch):
        loss_ce = F.cross_entropy(output, target)
        loss_kl = nn.KLDivLoss()(F.log_softmax(output/self.T, dim=1), 
            F.softmax(output_tch/self.T, dim=1)) * self.T * self.T
        loss_kd = loss_ce * (1. - self.alpha) + loss_kl * self.alpha
        # print("loss_ce", loss_ce, "loss_kl：", loss_kl)
        return loss_kd
