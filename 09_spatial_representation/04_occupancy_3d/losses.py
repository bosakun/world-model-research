from __future__ import annotations
import torch
import torch.nn.functional as F
def occupancy_loss(logits:torch.Tensor,target:torch.Tensor,positive_weight:float=12.,bce_weight:float=1.,dice_weight:float=1.):
    bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(positive_weight,device=logits.device))
    probability=logits.sigmoid(); intersection=(probability*target).sum(dim=(-3,-2,-1)); denominator=probability.sum(dim=(-3,-2,-1))+target.sum(dim=(-3,-2,-1))
    dice=1.-((2.*intersection+1e-6)/(denominator+1e-6)).mean(); return {"total":bce_weight*bce+dice_weight*dice,"bce":bce,"dice":dice}
