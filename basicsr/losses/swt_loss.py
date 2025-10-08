import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY
import basicsr.losses.SWT as SWT
import pywt
import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode

def downsample(image: torch.Tensor, scale_factor: float, interpolation: InterpolationMode = InterpolationMode.BICUBIC) -> torch.Tensor:
        h, w = image.shape[-2:]
        new_size = (int(h * scale_factor), int(w * scale_factor))
        return Resize(new_size, interpolation=interpolation, antialias=True)(image)

@LOSS_REGISTRY.register()
class SWTLoss(nn.Module):
    def __init__(self, loss_weight_l=0.01, loss_weight_h=0.01, alpha1=0.6, alpha2=0.3, alpha3=0.1, reduction='mean'):
        super(SWTLoss, self).__init__()
        self.loss_weight_l = loss_weight_l
        self.loss_weight_h = loss_weight_h
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        # wavelet = pywt.Wavelet('haar')
        wavelet = pywt.Wavelet('sym19')
            
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2*np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        # sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")
        sfm = SWT.SWTForward(2, filters, 'periodic').to("cuda")

        sr_img_y       = 16.0 + (pred[:,0:1,:,:]*65.481 + pred[:,1:2,:,:]*128.553 + pred[:,2:,:,:]*24.966)

        wavelet_sr  = sfm(sr_img_y)[0]

        L_sr   = wavelet_sr[:,0:1, :, :]
        H_sr   = wavelet_sr[:,1:, :, :]
        L_sr_2   = downsample(L_sr, 0.5)
        H_sr_2   = downsample(H_sr, 0.5)
        L_sr_4   = downsample(L_sr, 0.25)
        H_sr_4   = downsample(H_sr, 0.25)      

        hr_img_y       = 16.0 + (target[:,0:1,:,:]*65.481 + target[:,1:2,:,:]*128.553 + target[:,2:,:,:]*24.966)
     
        wavelet_hr     = sfm(hr_img_y)[0]

        L_hr   = wavelet_hr[:,0:1, :, :]
        H_hr   = wavelet_hr[:,1:, :, :]
        L_hr_2   = downsample(L_hr, 0.5)
        H_hr_2   = downsample(H_hr, 0.5)
        L_hr_4   = downsample(L_hr, 0.25)
        H_hr_4   = downsample(H_hr, 0.25)


        loss_L = self.loss_weight_l * (self.alpha1 * self.criterion(L_sr, L_hr) + self.alpha2 * self.criterion(L_sr_2, L_hr_2) + self.alpha3 * self.criterion(L_sr_4, L_hr_4))
        loss_H = self.loss_weight_h * (self.alpha1 * self.criterion(H_sr, H_hr) + self.alpha2 * self.criterion(H_sr_2, H_hr_2) + self.alpha3 * self.criterion(H_sr_4, H_hr_4))

        loss = loss_L + loss_H


        return loss
