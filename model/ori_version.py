# The Code Implementatio of MambaIR model for Real Image Denoising task
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial
from pytorch_wavelets import DWTForward, DWTInverse
import pywt as wt
import torchvision.utils as vutils
NEG_INF = -1000000



# 设置默认CUDA设备
# torch.cuda.set_device(3)


class Interaction_block(nn.Module):
    def __init__(self,num_feat):
        super(Interaction_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=num_feat,kernel_size=1,stride=1,padding=0) # dim for ir
        self.conv1_1 = nn.Conv2d(in_channels=3,out_channels=num_feat,kernel_size=1,stride=1,padding=0) # dim for vi_rain
        self.MP = nn.MaxPool2d(kernel_size=2,stride=2)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.AP = nn.AvgPool2d(kernel_size=2,stride=2)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.mp_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat//2,out_channels=num_feat//2,kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        self.ap_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat//2,out_channels=num_feat//2,kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        self.gmp_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat//2,out_channels=num_feat//2,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        self.gap_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_feat//2,out_channels=num_feat//2,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        self.upsamle1 = nn.ConvTranspose2d(in_channels=num_feat//2, out_channels=num_feat//2, kernel_size=4, stride=2, padding=1)
        self.upsamle2 = nn.ConvTranspose2d(in_channels=num_feat//2, out_channels=num_feat//2, kernel_size=4, stride=2, padding=1)
            
    def forward(self,ir,vi):
        # print(ir.shape)
        # print("x",vi_rain.shape)
        ir = self.conv1(ir) #  ir's dim the same as vi_rain
        # print(ir.shape)
        vi_rain = self.conv1_1(vi) # vi_rain's dim is 3
        C = ir.shape[1]
        ir_mp = ir[:, :C//2, :, :]
        # print("ir_mp",ir_mp.shape)
        ir_gmp = ir[:, C//2:, :, :]
        # print("ir_gmp",ir_gmp.shape)
        vi_rain_ap = vi_rain[:, :C//2, :, :]
        vi_rain_gap = vi_rain[:, C//2:, :, :]
        # print(vi_rain_gap.shape)
        ir_mp_conv = self.mp_conv(ir_mp)
        ir_gmp_conv = self.gmp_conv(ir_gmp)
        vi_rain_ap_conv = self.ap_conv(vi_rain_ap)
        vi_rain_gap_conv = self.gap_conv(vi_rain_gap)

        ir_mp_conv = self.upsamle2(ir_mp_conv)
        vi_rain_ap_conv = self.upsamle1(vi_rain_ap_conv)
        
        vi_rain_app = vi_rain_ap_conv * ir_mp
        ir_mp_app = ir_mp_conv * vi_rain_ap
        # print("vi_rain_app",vi_rain_app.shape)
        # print("vi_rain_gap_conv",vi_rain_gap_conv.shape)
        ir_gather  = ir_mp_app * ir_gmp_conv
        vi_gather  = vi_rain_app * vi_rain_gap_conv
        
        # ir_gather = self.upsamle2(ir_gather)
        # vi_gather = self.upsamle1(vi_gather)
        fuse_gather = torch.cat([ir_gather,vi_gather],dim=1)
        fuse_gather = rearrange(fuse_gather, "b c h w -> b (h w) c").contiguous()
        return fuse_gather
# class Interaction_block(nn.Module):
#     def __init__(self, num_feat):
#         super().__init__()
#         self.num_feat = num_feat
        
#         # 特征提取
#         self.conv_ir = nn.Sequential(
#             nn.Conv2d(3, num_feat, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         )
        
#         self.conv_vi = nn.Sequential(
#             nn.Conv2d(3, num_feat, 3, 1, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         )

#         # 多尺度特征提取 - 修改输出通道数
#         self.multiscale_ir = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(num_feat, num_feat, k, 1, k//2),
#                 nn.ReLU(True)
#             ) for k in [3, 5, 7]
#         ])
        
#         self.multiscale_vi = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(num_feat, num_feat, k, 1, k//2),
#                 nn.ReLU(True)
#             ) for k in [3, 5, 7]
#         ])
        
#         # 特征融合
#         self.fusion = nn.Conv2d(num_feat*2, num_feat, 1)
        
#         # 注意力机制
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_feat, num_feat//4, 1),
#             nn.ReLU(True),
#             nn.Conv2d(num_feat//4, num_feat, 1),
#             nn.Sigmoid()
#         )
        
#         # 空间注意力
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(num_feat*2, num_feat, 3, 1, 1),
#             nn.ReLU(True),
#             nn.Conv2d(num_feat, 1, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, ir, vi_rain):
#         # 1. 特征提取
#         ir_feat = self.conv_ir(ir)
#         vi_feat = self.conv_vi(vi_rain)
        
#         # 2. 多尺度特征提取
#         ir_multi = []
#         vi_multi = []
#         for ir_conv, vi_conv in zip(self.multiscale_ir, self.multiscale_vi):
#             ir_multi.append(ir_conv(ir_feat))
#             vi_multi.append(vi_conv(vi_feat))
        
#         # 合并多尺度特征
#         ir_multi = sum(ir_multi) / len(ir_multi)  # 平均融合
#         vi_multi = sum(vi_multi) / len(vi_multi)
        
#         # 3. 通道注意力
#         ir_channel = ir_feat * self.channel_attention(ir_feat)
#         vi_channel = vi_feat * self.channel_attention(vi_feat)
        
#         # 4. 空间注意力
#         spatial_feat = torch.cat([ir_feat, vi_feat], dim=1)
#         spatial_attn = self.spatial_attention(spatial_feat)
        
#         # 5. 特征增强
#         ir_enhanced = ir_channel * spatial_attn
#         vi_enhanced = vi_channel * spatial_attn
        
#         # 6. 特征融合
#         fused_feat = self.fusion(torch.cat([
#             ir_enhanced + ir_multi, 
#             vi_enhanced + vi_multi
#         ], dim=1))
        
#         # 7. 转换为序列形式
#         fused_gather = rearrange(fused_feat, "b c h w -> b (h w) c").contiguous()
        
#         return fused_gather


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4,squeeze_factor=16):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

class SS2D_HIGHFREQ(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # self.in_proj_HIGREQ = nn.Linear(self.d_model, self.d_inner , bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
                                              
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=6, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=6, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=6, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        # self.gg = self.d_inner//2
        self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_norm_HIGHFREQ = nn.LayerNorm(self.gg)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def diagonal_gather(self, tensor):
        # 取出矩阵所有反斜向的元素并拼接
        B, C, H, W = tensor.size()
        shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
        index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
        # 扩展索引以适应B和C维度
        expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        # 使用gather进行索引选择
        return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

    def diagonal_scatter(self, tensor_flat, original_shape):
        # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
        B, C, H, W = original_shape
        shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
        index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
        # 扩展索引以适应B和C维度
        expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        # 创建一个空的张量来存储反向散布的结果
        result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
        # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
        tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
        # 使用scatter_根据expanded_index将元素放回原位
        result_tensor.scatter_(3, expanded_index, tensor_reshaped)
        return result_tensor

    def forward_core(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        # print("SSDx.shape", x.shape, "SSDy.shape", y.shape)
        B, C, H, W = x.shape
        L = H * W
        K = 6


        h_x = x.view(B, -1, L)
        
        v_y = torch.transpose(y, dim0=2, dim1=3).contiguous().view(B, -1, L) #HL
        diag_z  = self.diagonal_gather(y)       

        highfreq = torch.stack([h_x, v_y, diag_z], dim=1).view(B, 3, -1, L) #x_hwwh

        xs = torch.cat([highfreq, torch.flip(highfreq, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        
 
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # print("dts.shape", dts.shape, "Bs.shape", Bs.shape, "Cs.shape", Cs.shape)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        
        inv_y = torch.flip(out_y[:, 3:6], dims=[-1]).view(B, 3, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        diag_y_out = self.diagonal_scatter(out_y[:,2:3],y.shape).contiguous().view(B, -1, L)
        invdiag_y_out = self.diagonal_scatter(inv_y[:,2:3],y.shape).contiguous().view(B, -1, L)
        
        
        # return HL_out, invHL_out, diag_y_out, invdiag_y_out # H D
        return out_y[:,0], inv_y[:,0], wh_y, invwh_y, diag_y_out, invdiag_y_out

    def forward(self, x: torch.Tensor,  y: torch.Tensor, k: torch.Tensor, **kwargs):
        # print("d_inner", self.d_inner)
        # print("pri_x.shape", x.shape, "pri_y.shape", y.shape)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        xz_y = self.in_proj(y)
        y, z = xz_y.chunk(2, dim=-1)

        xz_k = self.in_proj(k)
        k, z = xz_k.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 3, 1, 2).contiguous()
        k = k.permute(0, 3, 1, 2).contiguous()


        x = self.act(self.conv2d(x))
        y = self.act(self.conv2d(y))
        k = self.act(self.conv2d(k))

        y1, y2, y3, y4, y5, y6 = self.forward_core(x, y, k)

        assert y1.dtype == torch.float32
        y_h = y1 + y2
        y_v = y3 + y4
        y_D = y5 + y6

        y_h = torch.transpose(y_h, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # print("y_h.shape", y_h.shape)
        y_h = self.out_norm(y_h)
        # print("y_hh", y_h.shape)
        y_h = y_h * F.silu(z)
        out_h = self.out_proj(y_h)

        y_v = torch.transpose(y_v, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # print("y_v.shape", y_v.shape)
        y_v = self.out_norm(y_v)
        # print("y_vv", y_v.shape)    
        y_v = y_v * F.silu(z)
        out_v = self.out_proj(y_v)

        y_D = torch.transpose(y_D, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # print("y_D.shape", y_D.shape)
        y_D = self.out_norm(y_D)
        # print("y_DD", y_D.shape)
        y_D = y_D * F.silu(z)
        out_D = self.out_proj(y_D)

        if self.dropout is not None:
            out_h = self.dropout(out_h)
            out_v = self.dropout(out_v)
            out_D = self.dropout(out_D)
        return out_h, out_v, out_D
##########################################################################




########################################################################################################################

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        # print("SS2D_x.shape",x.shape)
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # print("SS2D_x_hwwh.shape",x_hwwh.shape)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        # print("SS2D_xs.shape",xs.shape)
#

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # print("SS2D_x_pri.shape",x.shape)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
            # print("out.shape",out.shape)
        return out




class WaveletDecomposition(nn.Module):
    def __init__(self, wave='haar', J=1):
        super(WaveletDecomposition, self).__init__()
        self.xfm = DWTForward(J=J, wave=wave,mode= 'zero')  # 初始化小波变换

    def forward(self, img):
        # 对每个通道分别进行小波分解
        # yl_g, yh_g = self.xfm(img[:, 0:1, :, :])  # G 通道
        # yl_r, yh_r = self.xfm(img[:, 1:2, :, :])  # R 通道
        # yl_b, yh_b = self.xfm(img[:, 2:3, :, :])  # B 通道
        # print("img",img.shape)
        img = img.permute(0, 3, 1, 2).contiguous()
        # print("img",img.shape)
        yl, yh = self.xfm(img)  # 合并三个通道的小波分解结果
        # print("yl",yl.shape)      
        # for i, h in enumerate(yh):
        #     print(f"yh[{i}] 的形状:", h.shape)
        # # 合并低频部分
        # # LL = torch.cat((yl_g, yl_r, yl_b), dim=1)  # 合并三个通道的低频系数

        # # 合并高频部分的三个方向（HL, LH, HH）
        # HL = torch.cat((yh_g[0][:, 0:1, :, :], yh_r[0][:, 0:1, :, :], yh_b[0][:, 0:1, :, :]), dim=1)
        # LH = torch.cat((yh_g[0][:, 1:2, :, :], yh_r[0][:, 1:2, :, :], yh_b[0][:, 1:2, :, :]), dim=1)
        # HH = torch.cat((yh_g[0][:, 2:3, :, :], yh_r[0][:, 2:3, :, :], yh_b[0][:, 2:3, :, :]), dim=1)
        LH = yh[0][:, :, 0:1, :, :].squeeze(2).permute(0, 2, 3, 1).contiguous()
        HL = yh[0][:, :, 1:2, :, :].squeeze(2).permute(0, 2, 3, 1).contiguous()
        HH = yh[0][:, :, 2:3, :, :].squeeze(2).permute(0, 2, 3, 1).contiguous()
        yl = yl.permute(0, 2, 3, 1).contiguous()
        # print("yl",yl.shape)
        # print("HL",HL.shape)
        return yl, LH, HL, HH


class WaveletReconstruction(nn.Module):
    def __init__(self, wave='haar'):
        super(WaveletReconstruction, self).__init__()
        self.ifm = DWTInverse(wave=wave)  # 初始化小波逆变换

    def forward(self, yl, HL, LH, HH):
        yl = yl.permute(0, 3, 1, 2).contiguous()
        HL = HL.permute(0, 3, 1, 2).contiguous()
        LH = LH.permute(0, 3, 1, 2).contiguous()
        HH = HH.permute(0, 3, 1, 2).contiguous()
        yh = torch.cat((HL.unsqueeze(2), LH.unsqueeze(2), HH.unsqueeze(2)), dim=2)  # 合并三个方向的高频分量
        # print("yh",yh.shape)
        # 进行逆小波变换
        yh_list = []
        yh_list.append(yh)
        # 合并三个通道的图像
        img = self.ifm((yl, yh_list)).permute(0, 2, 3, 1).contiguous()  # 逆小波变换
        # print("img",img.shape)
        return img

# 
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_11 = norm_layer(hidden_dim)
        self.ln_12 = norm_layer(hidden_dim)
        self.ln_13 = norm_layer(hidden_dim)

        self.ln_21 = norm_layer(hidden_dim)
        self.ln_22 = norm_layer(hidden_dim)
        self.ln_23 = norm_layer(hidden_dim)

        self.SWT = WaveletDecomposition().cuda()  # 初始化平稳小波变换
        self.ISWT = WaveletReconstruction().cuda()  # 初始化逆平稳小波变换

        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.self_attention_HIGHFREQ= SS2D_HIGHFREQ(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        
        self.drop_path = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.drop_path3 = DropPath(drop_path)
        self.drop_path4 = DropPath(drop_path)

        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale3 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale4 = nn.Parameter(torch.ones(hidden_dim))
        # self.skip_scale5 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale6 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale7 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale8 = nn.Parameter(torch.ones(hidden_dim))
        # self.skip_scale9 = nn.Parameter(torch.ones(hidden_dim))
        # self.conv_blk = CAB(hidden_dim)
        self.conv_blk1 = CAB(hidden_dim)
        self.conv_blk2 = CAB(hidden_dim)
        self.conv_blk3 = CAB(hidden_dim)
        # self.ln_2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, input, x_size):

        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]

        LL, LH, HL, HH = self.SWT(input)  # LH是垂直，HL是水平，HH是对角！

        LL_in = self.ln_1(LL)
        HL_in = self.ln_11(HL)
        LH_in = self.ln_12(LH)
        HH_in = self.ln_13(HH)



        LL_attn = LL_in*self.skip_scale + self.drop_path(self.self_attention(LL_in))
        LH_attn, HL_attn,HH_attn = self.self_attention_HIGHFREQ(LH_in,HL_in, HH_in)

        HL_VBD_attn = HL_in*self.skip_scale2 + self.drop_path2(HL_attn) 
        HH_HBD_attn = HH_in*self.skip_scale3 + self.drop_path3(HH_attn)
        LH_DBD_attn = LH_in*self.skip_scale4 + self.drop_path4(LH_attn)

        HL_VBD_attn = HL_VBD_attn*self.skip_scale6 + self.conv_blk1(self.ln_21(HL_VBD_attn).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        HH_HBD_attn = HH_HBD_attn*self.skip_scale7 + self.conv_blk2(self.ln_22(HH_HBD_attn).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        LH_DBD_attn = LH_DBD_attn*self.skip_scale8 + self.conv_blk3(self.ln_23(LH_DBD_attn).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        wavelet_feature = self.ISWT(LL_attn, LH_DBD_attn, HL_VBD_attn, HH_HBD_attn)


        # x = wavelet_feature*self.skip_scale5 + self.conv_blk(self.ln_2(wavelet_feature).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        wavelet_feature = wavelet_feature.view(B, -1, C).contiguous()

        return wavelet_feature


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
# class OverlapPatchEmbed(nn.Module):
#     def __init__(self, in_c=24, embed_dim=48, bias=False):
#         super(OverlapPatchEmbed, self).__init__()

#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
#     def forward(self, x):
#         x = self.proj(x)
#         x = rearrange(x, "b c h w -> b (h w) c").contiguous()
#         return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x



class WaveMamba(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 3, 5],
                 mlp_ratio=2.,
                 num_refinement_blocks=4,
                 drop_path_rate=0.,
                 bias=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(WaveMamba, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.interactions = Interaction_block(dim)
        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state,
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 3),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 3),
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.conv = nn.Conv2d(in_channels=1024, out_channels=3, kernel_size=1)       
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
    def forward(self, inp_rain,inp_img):
        _, _, H, W = inp_img.shape
        # print("inp_img shape:",inp_img.shape)
        # print(inp_img.device)

        inp_enc_level1 = self.interactions(inp_img,inp_rain)
        # inp_enc_level1 = self.patch_embed(inp_img)  # b,hw,c
        out_enc_level1 = inp_enc_level1
        # print("out_enc_level1_pri shape:",out_enc_level1.shape)
        
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])
        # print("out_enc_level1 shape:",out_enc_level1.shape)
        

        # inp_enc_level1 = out_enc_level1.view(1, 1024, 48, 1)  


        # output = self.upsample(inp_enc_level1)  


        # output = self.conv(output)
        # print("output shape:",output.shape)
        # return output
        out_enc_level1 = inp_enc_level1
        
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  # b, hw//4, 2c
 
        out_enc_level2 = inp_enc_level2
      
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  # b, hw//16, 4c

        out_enc_level3 = inp_enc_level3
   
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [H // 4, W // 4])

        inp_enc_level4 = self.down3_4(out_enc_level3, H // 4, W // 4)  # b, hw//64, 8c
  
        latent = inp_enc_level4
        # print("latenty   shape:",latent.shape)
        for layer in self.latent:
            latent = layer(latent, [H // 8, W // 8])
        # print("latent shape:",latent.shape)
        inp_dec_level3 = self.up4_3(latent, H // 8, W // 8)  # b, hw//16, 4c
       
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
     
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
    
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
    
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
  
        out_dec_level3 = inp_dec_level3
   
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, H // 4, W // 4)  # b, hw//4, 2c
   
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
  
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
    
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
      
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
     
        out_dec_level2 = inp_dec_level2
    
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
   
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
     
        out_dec_level1 = inp_dec_level1
        # print("out_dec_level1 shape:",out_dec_level1.shape)
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])
        # print("out_dec_level1 shape:",out_dec_level1.shape)
        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])
        # print("out_dec_level1 shape:",out_dec_level1.shape)
        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        # print("out_dec_level1 shape:",out_dec_level1.shape)
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img + inp_rain
        # print("out_dec_level1 shape:",out_dec_level1.shape)
        vutils.save_image(out_dec_level1, 'reconstructed_image.png', normalize=True)
        return out_dec_level1
