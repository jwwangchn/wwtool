import torch
import torch.nn as nn
import torch.nn.functional as F

def huber_loss(pred, target, sigma=1.0, reduction='mean'):
    return F.smooth_l1_loss(pred/sigma, target/sigma, reduction=reduction) * (sigma**2)