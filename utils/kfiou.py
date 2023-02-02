import functools
import torch
import torch.nn.functional as F

# Copyright (c) SJTU. All rights reserved.
import torch
#from mmdet.models.losses.utils import weighted_loss
from torch import nn

#from ..builder import ROTATED_LOSSES
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xy_wh_r_2_xy_sigma_old(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.
    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


#@weighted_loss
def kfiou_loss(pred,
               target,
               pred_decode=None,
               targets_decode=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.
    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.
    Returns:
        loss (torch.Tensor)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    
    #convert sigma_p&t to float32 or 'full-float' to support pytorch linear algebra functions
    Sigma_p = Sigma_p.type(torch.FloatTensor)
    Sigma_t = Sigma_t.type(torch.FloatTensor)

    #calculate the Vb of p & t
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()

    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    #always convert results back to float16 or 'half-float'
    KFIoU = KFIoU.type(torch.HalfTensor)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    #move all related tensors to GPU before finalisation
    xy_loss = xy_loss.to(device)
    kf_loss = kf_loss.to(device)

    loss = (xy_loss + kf_loss).clamp(0)
    #print(KFIoU.size())
    #print(loss.size())

    #always convert results back to float16 or 'half-float'
    #loss = loss.type(torch.HalfTensor).to(device)
    return KFIoU, loss


#@ROTATED_LOSSES.register_module()
class KFLoss(nn.Module):
    """Kalman filter based loss.
    Args:
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 fun='none',
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(KFLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['none', 'ln', 'exp']
        self.fun = fun
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                pred_decode=None,
                targets_decode=None,
                reduction_override=None,
                **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            pred_decode (torch.Tensor): Predicted decode bboxes.
            targets_decode (torch.Tensor): Corresponding gt decode bboxes.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        return kfiou_loss(
            pred,
            target,
            fun=self.fun,
            weight=weight,
            avg_factor=avg_factor,
            pred_decode=pred_decode,
            targets_decode=targets_decode,
            reduction=reduction,
            **kwargs) * self.loss_weight

    import torch
import torch.nn as nn

class KFiou(nn.Module):

    def __init__(self, fun='none',
                 beta=1.0 / 9.0,
                 eps=1e-6):
        super(KFiou, self).__init__()
        self.eps = eps
        self.beta = beta
        self.fun = fun

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        pred = pred.type(torch.float32)
        target = target.type(torch.float32)
        xy_p, Sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        xy_t, Sigma_t = self.xy_wh_r_2_xy_sigma(target)

        # Smooth-L1 norm
        diff = torch.abs(xy_p - xy_t)
        xy_loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                              diff - 0.5 * self.beta).sum(dim=-1)
        Vb_p = 4 * Sigma_p.det().clamp(1e-7).sqrt()
        Vb_t = 4 * Sigma_t.det().clamp(1e-7).sqrt()
        K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
        Sigma = Sigma_p - K.bmm(Sigma_p)
        Vb = 4 * Sigma.det().clamp(1e-7).sqrt()
        Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
        #Vb_p = torch.where(torch.isnan(Vb_p), torch.full_like(Vb_p, 0), Vb_p)
        #Vb_t = torch.where(torch.isnan(Vb_p), torch.full_like(Vb_t, 0), Vb_t)
        KFIoU = Vb / (Vb_p + Vb_t - Vb + self.eps) #Vb_p 有nan 

        if self.fun == 'ln':
            kf_loss = - torch.log(KFIoU + self.eps)
        elif self.fun == 'exp':
            kf_loss = torch.exp(1 - KFIoU) - 1
        else:
            kf_loss = 1 - KFIoU

        loss = (0.01 * xy_loss + kf_loss).clamp(1e-7)  #xy_loss 因为用的解码之后的 xy_loss 刚开始回比较大 所以直接缩小10倍
        KFIoU =  1 / (1 + torch.log1p(loss)) # use yolov iou 作为obj的rotion
        return loss, KFIoU

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.
        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).
        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r =  xywhr[..., 4]
        # 弧度制
        # r = (3.141592 * xywhr[..., 4]) / 180.0
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy.type(torch.float32), sigma.type(torch.float32)
