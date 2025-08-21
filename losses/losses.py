import torch
from torch import nn as nn


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pytorch_wavelets import DWTForward
# class PerceptualLoss(nn.Module):
#     def __init__(self, device=torch.device("cuda"), use_l1=True):
#         super().__init__()
#         # 加载 VGG19 并移动到指定设备
#         # self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device)
#         self.vgg = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features.to(device)
#         self.vgg.eval()
#         # 冻结参数
#         for param in self.vgg.parameters():
#             param.requires_grad = False
            
#         # 选择特征层（示例：relu1_2, relu2_2, relu3_2, relu4_2, relu5_2）
#         self.feature_layers = [2, 7, 12, 21, 30]
#         self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()
    
#     def forward(self, generated, target):
#         # 确保输入数据与模型在同一设备
#         # 扩展单通道到三通道（若输入为灰度图）
#         if generated.shape[1] == 1:
#             generated = generated.repeat(1, 3, 1, 1)
#         if target.shape[1] == 1:
#             target = target.repeat(1, 3, 1, 1)
        
#         # 提取特征
#         gen_features = []
#         target_features = []
#         x_gen = generated
#         x_target = target
#         for i, layer in enumerate(self.vgg):
#             x_gen = layer(x_gen)
#             x_target = layer(x_target)
#             if i in self.feature_layers:
#                 gen_features.append(x_gen)
#                 target_features.append(x_target)
        
#         # 计算损失
#         loss = 0.0
#         for gen, tar in zip(gen_features, target_features):
#             loss += self.criterion(gen, tar)
#         return loss / len(self.feature_layers)

# class Discriminator(nn.Module):
#     """ PatchGAN 判别器 """
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),  # 输入: [B,3,H,W]
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),         # 输出: [B,128,H/4,W/4]
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),        # 输出: [B,256,H/8,W/8]
#             nn.InstanceNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 1, 4, stride=1, padding=1)           # 输出: [B,1,H/8-3,W/8-3]
#         )

#     def forward(self, x):
#         return self.model(x)

# class GANLoss(nn.Module):
#     def __init__(self, use_lsgan=True):
#         super().__init__()
#         self.real_label = 1.0
#         self.fake_label = 0.0
#         if use_lsgan:
#             self.loss = nn.MSELoss()  # 最小二乘GAN
#         else:
#             self.loss = nn.BCEWithLogitsLoss()  # 传统GAN

#     def forward(self, prediction, is_real):
#         if is_real:
#             target = torch.tensor(self.real_label).expand_as(prediction).to(prediction.device)
#         else:
#             target = torch.tensor(self.fake_label).expand_as(prediction).to(prediction.device)
#         return self.loss(prediction, target)
    
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window


# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

#     C1 = 0.01**2
#     C2 = 0.03**2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

# def Contrast(img1, img2, window_size=11, channel=1):
#     window = create_window(window_size, channel)    
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#     mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)

#     sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq

#     return sigma1_sq, sigma2_sq

    
# class SSIMLoss(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)

#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)

#             self.window = window
#             self.channel = channel

#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# def ssim(img1, img2, window_size=11, size_average=True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
    
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
    
#     return _ssim(img1, img2, window, window_size, channel, size_average)



# class L_color(nn.Module):
#     def __init__(self):
#         super(L_color, self).__init__()

#     def forward(self, image_visible, image_fused):
#         ycbcr_visible = self.rgb_to_ycbcr(image_visible)
#         ycbcr_fused = self.rgb_to_ycbcr(image_fused)

#         cb_visible = ycbcr_visible[:, 1, :, :]
#         cr_visible = ycbcr_visible[:, 2, :, :]
#         cb_fused = ycbcr_fused[:, 1, :, :]
#         cr_fused = ycbcr_fused[:, 2, :, :]

#         loss_cb = F.l1_loss(cb_visible, cb_fused)
#         loss_cr = F.l1_loss(cr_visible, cr_fused)

#         loss_color = loss_cb + loss_cr
#         return loss_color

#     def rgb_to_ycbcr(self, image):
#         r = image[:, 0, :, :]
#         g = image[:, 1, :, :]
#         b = image[:, 2, :, :]

#         y = 0.299 * r + 0.587 * g + 0.114 * b
#         cb = -0.168736 * r - 0.331264 * g + 0.5 * b
#         cr = 0.5 * r - 0.418688 * g - 0.081312 * b

#         ycbcr_image = torch.stack((y, cb, cr), dim=1)
#         return ycbcr_image
    


# class L_SSIM(nn.Module):
#     def __init__(self):
#         super(L_SSIM, self).__init__()
#         self.sobelconv = Sobelxy()

#     def forward(self, image_A, image_B, image_fused):
#         gradient_A = self.sobelconv(image_A)
#         gradient_B = self.sobelconv(image_B)
#         weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
#         return Loss_SSIM


# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                    [0, 0, 0],
#                    [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

#     def forward(self, x):
#         sobelx = F.conv2d(x, self.weightx, padding=1)
#         sobely = F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx) + torch.abs(sobely)   
    



# class L_color(nn.Module):

#     def __init__(self):
#         super(L_color, self).__init__()

#     def forward(self, x):
#         b, c, h, w = x.shape

#         mean_rgb = torch.mean(x, [2, 3], keepdim=True)
#         mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
#         Drg = torch.pow(mr - mg, 2)
#         Drb = torch.pow(mr - mb, 2)
#         Dgb = torch.pow(mb - mg, 2)
#         k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
#         return k


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


# class L_SSIM(nn.Module):
#     def __init__(self):
#         super(L_SSIM, self).__init__()
#         self.sobelconv = Sobelxy()

#     def forward(self, image_A, image_B, image_fused):
#         # print('b',image_B.shape)
#         # print('a',image_A.shape)
#         gradient_A = self.sobelconv(image_A)
#         # C = image_B.shape[1]
        
#         # imgb = image_B[:, :1, :, :]
#         # print(imgb.shape)
#         gradient_B = self.sobelconv(image_B)
#         weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
#         Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
#         return Loss_SSIM


# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                    [0, 0, 0],
#                    [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

#     def forward(self, x):
#         sobelx = F.conv2d(x, self.weightx, padding=1)
#         sobely = F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx) + torch.abs(sobely)


# class L_Intensity(nn.Module):
#     def __init__(self):
#         super(L_Intensity, self).__init__()

#     def forward(self, image_A, image_B, image_fused):
#         # print(image_A.shape,image_B.shape,image_fused.shape)
#         intensity_joint = torch.max(image_A, image_B)
#         # print(intensity_joint.shape)
#         Loss_intensity = F.l1_loss(image_fused, intensity_joint)
#         return Loss_intensity


# class L_color2(nn.Module):
#     def __init__(self):
#         super(L_color2, self).__init__()

#     def forward(self, image_visible, image_fused):
#         ycbcr_visible = self.rgb_to_ycbcr(image_visible)
#         ycbcr_fused = self.rgb_to_ycbcr(image_fused)

#         cb_visible = ycbcr_visible[:, 1, :, :]
#         cr_visible = ycbcr_visible[:, 2, :, :]
#         cb_fused = ycbcr_fused[:, 1, :, :]
#         cr_fused = ycbcr_fused[:, 2, :, :]

#         loss_cb = F.l1_loss(cb_visible, cb_fused)
#         loss_cr = F.l1_loss(cr_visible, cr_fused)

#         loss_color = loss_cb + loss_cr
#         return loss_color

#     def rgb_to_ycbcr(self, image):
#         r = image[:, 0, :, :]
#         g = image[:, 1, :, :]
#         b = image[:, 2, :, :]

#         y = 0.299 * r + 0.587 * g + 0.114 * b
#         cb = -0.168736 * r - 0.331264 * g + 0.5 * b
#         cr = 0.5 * r - 0.418688 * g - 0.081312 * b

#         ycbcr_image = torch.stack((y, cb, cr), dim=1)
#         return ycbcr_image


# # class fusion_loss_vif(nn.Module):
# #     def __init__(self, device='cuda'):
# #         super().__init__()
# #         self.L_Grad = L_Grad()
# #         self.L_Inten = L_Intensity()
# #         self.L_SSIM = L_SSIM()
# #         self.loss_func_color = L_color2()

# #     def wavelet_loss(self, pred, target):
# #         # 确保输入是4维张量 [B,C,H,W]
# #         if len(pred.shape) == 3:
# #             pred = pred.unsqueeze(0)
# #         if len(target.shape) == 3:
# #             target = target.unsqueeze(0)
            
# #         # 小波分解
# #         pred_ll, pred_high = self.swt(pred)
# #         target_ll, target_high = self.swt(target)
        
# #         # 低频损失
# #         ll_loss = F.l1_loss(pred_ll, target_ll)
        
# #         # 高频损失
# #         high_loss = sum([F.l1_loss(p, t) for p, t in zip(pred_high[0], target_high[0])]) / 3
        
# #         return ll_loss, high_loss
# #     def forward(self, image_RGB, image_B, image_fused_RGB):
# #         # 基础损失计算
# #         image_A = torch.mean(image_RGB, dim=1, keepdim=True)
# #         image_B = torch.mean(image_B, dim=1, keepdim=True)
# #         image_fused = torch.mean(image_fused_RGB, dim=1, keepdim=True)
        
# #         # 1. 基本融合损失
# #         loss_l1 = self.L_Inten(image_A, image_B, image_fused)
# #         loss_gradient = self.L_Grad(image_A, image_B, image_fused)
# #         loss_SSIM = 1 - self.L_SSIM(image_A, image_B, image_fused)
# #         loss_color = self.loss_func_color(image_RGB, image_fused_RGB)
        
# #         # 2. 频域损失
# #         ll_loss, high_loss = self.wavelet_loss(image_fused_RGB, image_RGB)
        
# #         # 动态权重
# #         total_loss = (
# #             5 * loss_l1 +           # 亮度损失
# #             5 * loss_gradient +     # 梯度损失
# #             5 * loss_SSIM +        # 结构损失
# #             3 * loss_color +       # 色彩损失
# #             2 * ll_loss +          # 低频损失（雾）
# #             5 * high_loss          # 高频损失（雨雪）
# #         )
        
# #         return total_loss

# class fusion_loss_vif(nn.Module):
#     def __init__(self, device='cuda'):
#         super().__init__()
#         self.L_Grad = L_Grad()
#         self.L_Inten = L_Intensity()
#         self.L_SSIM = L_SSIM()
#         self.loss_func_color = L_color2()
        
#         # 感知损失
#         self.perceptual = PerceptualLoss(device=device)
        
#     def forward(self, image_RGB, image_B, image_fused_RGB):
#         # 基础损失计算
#         image_A = torch.mean(image_RGB, dim=1, keepdim=True)
#         image_B = torch.mean(image_B, dim=1, keepdim=True)
#         image_fused = torch.mean(image_fused_RGB, dim=1, keepdim=True)
        
#         # 1. 亮度损失
#         loss_intensity = self.L_Inten(image_A, image_B, image_fused)
        
#         # 2. 梯度损失
#         loss_gradient = self.L_Grad(image_A, image_B, image_fused)
        
#         # 3. 结构损失
#         loss_SSIM = 1 - self.L_SSIM(image_A, image_B, image_fused)
        
#         # 4. 颜色损失
#         loss_color = self.loss_func_color(image_RGB, image_fused_RGB)
        
#         # 5. 感知损失
#         loss_perceptual = self.perceptual(image_fused_RGB, image_RGB)
        
#         # 计算总损失 - 确保是标量
#         total_loss = (
#             5.0 * loss_intensity +    # 转换为标量torchinfo
#             5.0 * loss_gradient +     # 转换为标量
#             5.0 * loss_SSIM +        # 转换为标量
#             3.0 * loss_color +       # 转换为标量
#             0.1 * loss_perceptual   # 转换为标量
#         )
        
#         return total_loss


# class fusion_loss_ir(nn.Module):
#     def __init__(self):
#         super(fusion_loss_ir, self).__init__()
#         self.L_Inten = L_Intensity()
#         self.L_SSIM = L_SSIM()
#         self.L_per = PerceptualLoss()
#         # self.Gandiscriminator = Discriminator()
#         # self.L_gan = GANLoss()
#     def forward(self, image_A, image_B, image_fused_ir):
#         c = image_A
#         loss_per = 10 * self.L_per(image_fused_ir, c)
#         image_A = torch.mean(image_A, dim=1, keepdim=True)
#         image_B = torch.mean(image_B, dim=1, keepdim=True)
#         image_fused_ir = torch.mean(image_fused_ir, dim=1, keepdim=True)
#         loss_l1 = 10 * self.L_Inten(image_A, image_B, image_fused_ir)
#         loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused_ir))

#         fusion_loss_ir = loss_l1   + loss_SSIM  + loss_per
#         return fusion_loss_ir

# class fusion_loss_vif_ori(nn.Module):
#     def __init__(self):
#         super(fusion_loss_vif_ori, self).__init__()
#         self.L_Grad = L_Grad()
#         self.L_Inten = L_Intensity()
#         self.L_SSIM = L_SSIM()
#         self.loss_func_color = L_color2()
        
#         # print(1)


#     def forward(self, image_RGB, image_B, image_fused_RGB):  # image_A = VIS image_B = IR
#         image_A = torch.mean(image_RGB, dim=1, keepdim=True)
#         image_B = torch.mean(image_B, dim=1, keepdim=True)
#         image_fused = torch.mean(image_fused_RGB, dim=1, keepdim=True)
#         loss_l1 = 10 * self.L_Inten(image_A, image_B, image_fused)
#         loss_gradient = 10 * self.L_Grad(image_A, image_B, image_fused)
#         loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
#         loss_color = 10 * self.loss_func_color(image_RGB, image_fused_RGB)
#         fusion_loss = loss_l1 + loss_gradient + loss_SSIM + loss_color 
#         # return fusion_loss, loss_gradient, loss_l1, loss_SSIM, loss_color
#         return fusion_loss
    
#     def cosine_similarity(self,text_features, image_features):
#         text_features = F.normalize(text_features, dim=-1)
#         image_features = F.normalize(image_features, dim=-1)

#         similarity = torch.sum(text_features * image_features, dim=-1)
#         loss = 1 - similarity.mean()

#         return loss



#2025.5.26-因为去雨去雪指标不行！
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pytorch_wavelets import DWTForward, DWTInverse
import torchvision.utils as vutils

"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

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

class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B) + 1e-6)
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B) + 1e-6)
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

class PerceptualLoss(nn.Module):
    def __init__(self, device=torch.device("cuda"), use_l1=True):
        super().__init__()
        self.vgg = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features.to(device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.feature_layers = [2, 7, 12, 21, 30]
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()
    
    def forward(self, generated, target):
        if generated.shape[1] == 1:
            generated = generated.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        gen_features = []
        target_features = []
        x_gen = generated
        x_target = target
        for i, layer in enumerate(self.vgg):
            x_gen = layer(x_gen)
            x_target = layer(x_target)
            if i in self.feature_layers:
                gen_features.append(x_gen)
                target_features.append(x_target)
        
        loss = 0.0
        for gen, tar in zip(gen_features, target_features):
            loss += self.criterion(gen, tar)
        return loss / len(self.feature_layers)

class WaveletLoss(nn.Module):
    def __init__(self, wave='haar', mode='zero',device=torch.device("cuda")):
        super().__init__()
        self.swt = DWTForward(J=1, wave=wave, mode=mode).to(device)
        
    def forward(self, pred, target):
        y_l, y_h = self.swt(pred)
        x_l, x_h = self.swt(target)
        
        ll_loss = F.l1_loss(y_l, x_l)
        lh_loss = F.l1_loss(y_h[0][:, :, 0, :, :], x_h[0][:, :, 0, :, :])
        hl_loss = F.l1_loss(y_h[0][:, :, 1, :, :], x_h[0][:, :, 1, :, :])
        hh_loss = F.l1_loss(y_h[0][:, :, 2, :, :], x_h[0][:, :, 2, :, :])
        high_loss = (lh_loss + hl_loss + hh_loss) / 3
        
        loss = ll_loss + high_loss
        return loss



class fusion_loss_vif(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.loss_func_color = L_color2()
        self.perceptual = PerceptualLoss(device=device)
        self.wavelet_loss = WaveletLoss()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的权重

        
    def forward(self, image_RGB, image_B, image_fused_RGB):
        image_A = torch.mean(image_RGB, dim=1, keepdim=True)
        image_B = torch.mean(image_B, dim=1, keepdim=True)
        image_fused = torch.mean(image_fused_RGB, dim=1, keepdim=True)
        
        loss_intensity = self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 1 - self.L_SSIM(image_A, image_B, image_fused)
        loss_color = self.loss_func_color(image_RGB, image_fused_RGB)
        loss_perceptual = self.perceptual(image_fused_RGB, image_RGB)
        wavelet_loss = self.wavelet_loss(image_fused_RGB, image_RGB)
        
 

        total_loss = (
            5.0 * loss_intensity +
            5.0 * loss_gradient +
            5.0 * loss_SSIM +
            3.0 * loss_color +
            0.1 * loss_perceptual +
            self.alpha * wavelet_loss  # 使用可学习的权重
        )
        
        return total_loss
