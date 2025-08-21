import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_ssim import ssim
# from basicsr.models.losses.losses.
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from .loss_ssim import ssim

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


class L_color2(nn.Module):
    def __init__(self):
        super(L_color2, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image
    
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d

class WaveletAwareLoss(nn.Module):
    def __init__(self, device):
        super(WaveletAwareLoss, self).__init__()
        self.device = device
        self.swt = DWTForward(J=1, mode='haar').to(device)
        self.l1 = nn.L1Loss()
        self.sobelconv = Sobelxy().to(device)
        
    def wavelet_loss(self, pred, target):
        # 小波分解
        pred_ll, pred_high = self.swt(pred)
        target_ll, target_high = self.swt(target)
        
        # 低频损失(雾的处理)
        ll_loss = self.l1(pred_ll, target_ll)
        
        # 高频损失(雨雪的处理)
        lh_loss = self.l1(pred_high[0][:,0], target_high[0][:,0])
        hl_loss = self.l1(pred_high[0][:,1], target_high[0][:,1])
        hh_loss = self.l1(pred_high[0][:,2], target_high[0][:,2])
        
        high_loss = (lh_loss + hl_loss + hh_loss) / 3.0
        
        return ll_loss, high_loss
    
    def forward(self, vi_rgb, ir, fused_rgb):
        # 基本损失
        intensity_loss = self.L_Inten(vi_rgb, ir, fused_rgb)
        gradient_loss = self.L_Grad(vi_rgb, ir, fused_rgb)
        
        # 小波域损失
        ll_loss, high_loss = self.wavelet_loss(fused_rgb, vi_rgb)
        
        # 结构损失
        ssim_loss = 1 - ssim(fused_rgb, vi_rgb)
        
        total_loss = (10 * intensity_loss + 
                     5 * gradient_loss + 
                     2 * ll_loss +  # 雾的权重较小
                     5 * high_loss +  # 雨雪的权重较大
                     3 * ssim_loss)
                     
        return total_loss



class fusion_loss_vif(nn.Module):
    def __init__(self, device):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad().to(device)
        self.L_Inten = L_Intensity().to(device)
        self.L_SSIM = L_SSIM().to(device)
        self.loss_func_color = L_color2().to(device)
        self.fre_loss = WaveletAwareLoss().to(device)
        # self.Low = L_exp().to(device)
        # print(1)

    def forward(self, image_RGB, image_B, image_fused_RGB):  # image_A = VIS image_B = IR
        image_A = torch.mean(image_RGB, dim=1, keepdim=True)
        image_fused = torch.mean(image_fused_RGB, dim=1, keepdim=True)
        loss_l1 = 10 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 10 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        loss_color = 10 * self.loss_func_color(image_RGB, image_fused_RGB)
        freq_loss = 10 * self.fre_loss(image_RGB, image_B, image_fused_RGB)
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_color + freq_loss
        # lowlight_loss = self.Low(image_fused_RGB)
        return fusion_loss

