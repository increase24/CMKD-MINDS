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
        return loss_kd

# def loss_cls_kd_fn(outputs, labels, teacher_outputs, params):
#     """
#     Compute the knowledge-distillation (KD) loss given outputs, labels.
#     "Hyperparameters": temperature and alpha
#     NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
#     and student expects the input tensor to be log probabilities! See Issue #2
#     """
#     alpha = params.alpha
#     T = params.temperature
#     KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
#                              F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
#               F.cross_entropy(outputs, labels) * (1. - alpha)

#     return KD_loss