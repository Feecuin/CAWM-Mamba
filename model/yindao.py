
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
from torchvision.ops import DeformConv2d
# from xformers.ops import memory_efficient_attention
# 设置默认CUDA设备
# torch.cuda.set_device(1,2,3)




class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class wavelet(nn.Module):
    def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
        w = wt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
        dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                   dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
        rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                   rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

        rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

        return dec_filters, rec_filters

    def wavelet_transform(x, filters):
        b, c, h, w = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
        x = x.reshape(b, c, 4, h // 2, w // 2)
        return x

    def inverse_wavelet_transform(x, filters):
        b, c, _, h_half, w_half = x.shape
        pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
        x = x.reshape(b, c * 4, h_half, w_half)
        x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
        return x


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

        img = img.permute(0, 3, 1, 2).contiguous()

        yl, yh = self.xfm(img)  # 合并三个通道的小波分解结果
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

        yh_list = []
        yh_list.append(yh)

        img = self.ifm((yl, yh_list)).permute(0, 2, 3, 1).contiguous()  # 逆小波变换

        return img


class SimpleSRModule(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):  # 输入通道数=LL(3) + LH(3)+HL(3)+HH(3)=12  [B, C, H, W]
        super(SimpleSRModule, self).__init__()
        self.upscale_factor = upscale_factor
        
        # 输入通道数调整为合并所有小波分量
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # 上采样部分
        self.upsample_conv = nn.Conv2d(32, out_channels * (upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, LL_IR, LH_IR, HL_IR, HH_IR):
        
        # 将低频和高频分量沿通道维度拼接
        x = torch.cat([LL_IR, LH_IR, HL_IR, HH_IR], dim=1)  # 通道数=3+3+3+3=12
        # print(x.shape,LL_IR.shape,LH_IR.shape,HH_IR.shape,'xxx')
        # 特征提取
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # 上采样
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        
        return x


class WaveletEnhancementBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.offset_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * 3 * 3,  # 3x3卷积核需要2 * 3 * 3=18通道的偏移量
            kernel_size=3,
            padding=1
        )
        # 可变形卷积（适应不规则边缘）
        self.deform_conv = DeformConv2d(
            in_channels, in_channels, kernel_size=3, padding=1
        )
        
        # 通道注意力（聚焦重要频段）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1),  # 防止通道过小
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 高频残差学习
        self.res_conv = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x_high):
        # 可变形卷积提取几何自适应特征
        offset = self.offset_conv(x_high)  # [B, 18, H, W]
        deform_feat = self.deform_conv(x_high, offset)
        
        # 通道注意力加权
        weight = self.ca(deform_feat)
        enhanced_high = deform_feat * weight
        
        # 残差连接保留原始高频信息
        res = self.res_conv(x_high)
        return enhanced_high + res

class WaveletReconstructionNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 高频增强模块（处理LH/HL/HH）
        self.lh_enhance = WaveletEnhancementBlock(in_channels)
        self.hl_enhance = WaveletEnhancementBlock(in_channels)
        self.hh_enhance = WaveletEnhancementBlock(in_channels)
        
        # 低频通道保持
        self.ll_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        # # 跨频段融合
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(in_channels * 4, in_channels, 1),  # LL + LH + HL + HH
        #     nn.ReLU()
        # )
    def forward(self, ll, lh, hl, hh):
        # 低频处理
        ll_feat = self.ll_conv(ll)
        # print("lht",lh.shape)
        # 高频增强
        lh_feat = self.lh_enhance(lh)
        hl_feat = self.hl_enhance(hl)
        hh_feat = self.hh_enhance(hh)
        # print("lh_feat",lh_feat.shape)
        # print("hl_feat",hl_feat.shape)
        # print("hh_feat",hh_feat.shape)
        # 拼接多频段特征
        # restored = self.iwt(ll_feat, lh_feat, hl_feat, hh_feat)
        # fused = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)
        # print("fused",fused.shape)
        # 重构并输出
        return ll_feat, lh_feat , hl_feat, hh_feat


def window_partition(x, window_size):
    """ 将特征图划分为不重叠的局部窗口 """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, H//k, W//k, C, k, k]
    windows = windows.view(-1, C, window_size, window_size)  # [B*num_win, C, k, k]
    return windows

def window_reverse(windows, window_size, H_pad, W_pad):
    """ 将局部窗口合并回完整特征图 
    Args:
        windows: (B*num_win, C, window_size, window_size)
        H_pad: 填充后的高度
        W_pad: 填充后的宽度
    """
    num_windows = (H_pad // window_size) * (W_pad // window_size)
    B = windows.shape[0] // num_windows
    C = windows.shape[1]
    x = windows.view(B, H_pad // window_size, W_pad // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H_pad, W_pad)
    return x

class CrossAttentionLLFusion(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, base_window_size=16):
        super().__init__()
        self.base_window_size = base_window_size
        
        # 统一投影维度
        self.proj_ir = nn.Conv2d(in_channels, embed_dim, 1)  # Query投影
        self.proj_vi = nn.Conv2d(in_channels, embed_dim, 1)  # Key/Value投影
        
        # 动态注意力门控
        self.attn_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//4, 1, 1),
            nn.Sigmoid()
        )
        
        # 输出转换（通道数修正）
        self.out_conv = nn.Conv2d(embed_dim, in_channels, 1)  # 输入通道修正为embed_dim

    def forward(self, ll_ir, ll_vi):
        B, C, H, W = ll_ir.shape
        window_size = self.base_window_size
        
        # 动态窗口调整策略
        valid_sizes = [k for k in [8, 16, 32] if (H % k == 0) and (W % k == 0)]
        window_size = max(valid_sizes) if valid_sizes else self.base_window_size
        
        # 智能填充处理
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        ll_ir = F.pad(ll_ir, (0, pad_w, 0, pad_h))
        ll_vi = F.pad(ll_vi, (0, pad_w, 0, pad_h))
        H_pad, W_pad = ll_ir.shape[2], ll_ir.shape[3]

        # 投影操作
        q_ir = self.proj_ir(ll_ir)  # [B, ED, H_pad, W_pad]
        k_vi = self.proj_vi(ll_vi)  # [B, ED, H_pad, W_pad]
        v_vi = k_vi.clone()

        # 窗口划分
        q_windows = window_partition(q_ir, window_size)  # [B*N, ED, k, k]
        k_windows = window_partition(k_vi, window_size)
        v_windows = window_partition(v_vi, window_size)

        # 注意力计算
        q = q_windows.view(-1, window_size**2, self.proj_ir.out_channels)  # [B*N, k^2, ED]
        k = k_windows.view(-1, window_size**2, self.proj_vi.out_channels)  # [B*N, k^2, ED]
        energy = torch.bmm(q, k.transpose(1,2))  # [B*N, k^2, k^2]
        
        # 注意力门控
        attn_mask = self.attn_gate(k_vi)
        attn_mask_windows = window_partition(attn_mask, window_size)  # [B*N, 1, k, k]
        energy = energy * attn_mask_windows.view(-1, 1, window_size**2)
        
        # 注意力权重
        attention = F.softmax(energy, dim=-1)
        
        # 值聚合
        v = v_windows.view(-1, window_size**2, self.proj_vi.out_channels)
        out = torch.bmm(attention, v)  # [B*N, k^2, ED]
        out = out.view(-1, window_size, window_size, self.proj_vi.out_channels)
        out = out.permute(0, 3, 1, 2)  # [B*N, ED, k, k]

        # 窗口合并
        out = window_reverse(out, window_size, H_pad, W_pad)  # [B, ED, H_pad, W_pad]
        
        # 残差连接与输出
        out = self.out_conv(out) + ll_ir  # 通道维度对齐
        out = out[:, :, :H, :W]  # 裁剪填充部分
        return out

class IRGuidanceBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        # 结构引导模块 
        self.structure_guide = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )
        
        # 自适应门控
        self.gate = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x_vi, x_ir):
        """
        Args:
            x_vi: 可见光特征 [B,C,H,W]
            x_ir: 红外特征 [B,C,H,W]
        Returns:
            guided_feat: 引导后的特征 [B,C,H,W]
        """
        # 特征拼接
        feat_cat = torch.cat([x_vi, x_ir], dim=1)
        
        # 结构引导
        struct_feat = self.structure_guide(feat_cat)
        
        # 自适应门控 - 控制红外信息的影响程度
        gate = self.gate(feat_cat)
        
        # 可见光特征增强
        enhanced = self.enhance(x_vi)
        
        # 引导特征融合 - 保持可见光特征主导
        guided_feat = x_vi + gate * struct_feat + 0.1 * enhanced
        
        return guided_feat
        
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
        self.ln_11 = norm_layer(hidden_dim)
        self.ln_12 = norm_layer(hidden_dim)
        self.ln_13 = norm_layer(hidden_dim)

        self.SWT = WaveletDecomposition().cuda()  # 初始化平稳小波变换
        self.ISWT = WaveletReconstruction().cuda()  # 初始化逆平稳小波变换

        self.self_attention_HIGHFREQ= SS2D_HIGHFREQ(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.self_attention_normal = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.drop_path3 = DropPath(drop_path)
        self.drop_path4 = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale3 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale4 = nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale5 = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)    
 
        # self.GELU = nn.GELU()

        self.ir_guide = IRGuidanceBlock(hidden_dim)

        self.irr_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )
    def forward(self, input_VI, input_IR, x_size):

        B, L, C = input_VI.shape
        input_VI = input_VI.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        input_IR = input_IR.view(B, *x_size, C).contiguous()  # [B,H,W,C]




# 调整维度顺序：NHWC -> NCHW
        LL_VI_c = input_VI.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        LL_IR_c = input_IR.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]


        guided_feat = self.ir_guide(LL_VI_c, LL_IR_c)
        guided_feat = guided_feat.permute(0,2,3,1).contiguous()
        LL_VI, LH_VI, HL_VI, HH_VI = self.SWT(guided_feat)  # LH是垂直，HL是水平，HH是对角！
        # LL_IR, LH_IR, HL_IR, HH_IR = self.SWT(input_IR)  # LH是垂直，HL是水平，HH是对角！
        LH_VI_in = self.ln_11(LH_VI)
        HL_VI_in = self.ln_12(HL_VI)
        HH_VI_in = self.ln_13(HH_VI)
        LL_VI_in = self.ln_13(LL_VI)
        LL_VI_attn = self.self_attention_normal(LL_VI_in)
        LL_VI_attn = LL_VI_attn*self.skip_scale + self.drop_path(LL_VI_attn)        
        LH_VI_attn, HL_VI_attn,HH_VI_attn = self.self_attention_HIGHFREQ(LH_VI_in,HL_VI_in, HH_VI_in)
        LH_VI_attn = LH_VI_attn*self.skip_scale2 + self.drop_path2(LH_VI_attn) 
        HL_VI_attn = HL_VI_attn*self.skip_scale3 + self.drop_path3(HL_VI_attn)
        HH_VI_attn = HH_VI_attn*self.skip_scale4 + self.drop_path4(HH_VI_attn)        
        wavelet_feature = self.ISWT(LL_VI_attn, LH_VI_attn, HL_VI_attn, HH_VI_attn)      
        x_VI = wavelet_feature*self.skip_scale5 + self.conv_blk(self.ln_2(wavelet_feature).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x_VI = x_VI.view(B, -1, C).contiguous()

        input_IR = input_IR.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        input_IR = self.irr_block(input_IR)
        input_IR = input_IR.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        input_IR = input_IR.view(B, -1, C).contiguous()
        # wavelet_feature_IR = self.ISWT( LL_IR_c, LH_IR, HL_IR, HH_IR)
        # wavelet_feature_IR = wavelet_feature_IR.permute(0, 2, 3, 1).contiguous()
        # LL_IR_c = wavelet_feature_IR.view(B, -1, C).contiguous()    
        return x_VI, input_IR
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


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
                 num_blocks=[2, 3, 3, 4],
                 mlp_ratio=2,
                 dim=48,
                 num_refinement_blocks=4,
                 drop_path_rate=0.,
                 bias=False,
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):
        


        super(WaveMamba, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=48,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state,
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(48)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(96),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(96)  ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=192,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.down3_4 = Downsample(192)  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(384),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 3),
            )
            for i in range(num_blocks[3])])

        self.up4_3 = Upsample(384)  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(384, 192, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(192),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(192)  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(192, 96, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(96),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(96)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=96,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 1),
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            VSSBlock(
                hidden_dim=96,
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

        self.output = nn.Conv2d(96, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, inp_img_VI, inp_img_IR):
        _, _, H, W = inp_img_VI.shape
 
        inp_enc_level1_one = self.patch_embed(inp_img_VI)  # b, hw, c
        inp_enc_level1_two = self.patch_embed(inp_img_IR)  # b, hw, c
        out_enc_level1_one = inp_enc_level1_one
        out_enc_level1_two = inp_enc_level1_two
        
        for layer in self.encoder_level1:
            out_enc_level1_one, out_enc_level1_two = layer(out_enc_level1_one, out_enc_level1_two,  [H, W])

        inp_enc_level2_one = self.down1_2(out_enc_level1_one, H, W)  # b, hw//4, 2c
        inp_enc_level2_two = self.down1_2(out_enc_level1_two, H, W)  # b, hw//4, 2c 
        out_enc_level2_one = inp_enc_level2_one
        out_enc_level2_two = inp_enc_level2_two
      
        for layer in self.encoder_level2:
            out_enc_level2_one, out_enc_level2_two = layer(out_enc_level2_one, out_enc_level2_two, [H // 2, W // 2])

        inp_enc_level3_one = self.down2_3(out_enc_level2_one,  H // 2, W // 2)  # b, hw//16, 4c
        inp_enc_level3_two = self.down2_3(out_enc_level2_two,  H // 2, W // 2)  # b, hw//16, 4c
        out_enc_level3_one, out_enc_level3_two = inp_enc_level3_one, inp_enc_level3_two
   
        for layer in self.encoder_level3:
            out_enc_level3_one, out_enc_level3_two = layer(out_enc_level3_one, out_enc_level3_two, [H // 4, W // 4])

        inp_enc_level4_one = self.down3_4(out_enc_level3_one, H // 4, W // 4)  # b, hw//64, 8c
        inp_enc_level4_two = self.down3_4(out_enc_level3_two, H // 4, W // 4)  # b, hw//64, 8c  
        latent_one, latent_two = inp_enc_level4_one, inp_enc_level4_two

        for layer in self.latent:
            latent_one, latent_two = layer(latent_one, latent_two, [H // 8, W // 8])

        inp_dec_level3_one = self.up4_3(latent_one, H // 8, W // 8)  # b, hw//16, 4c
        inp_dec_level3_two = self.up4_3(latent_two, H // 8, W // 8)  # b, hw//16, 4c      
        inp_dec_level3_one = torch.cat([inp_dec_level3_one, out_enc_level3_one], 2)
        inp_dec_level3_two = torch.cat([inp_dec_level3_two, out_enc_level3_two], 2)
     

     
        inp_dec_level3_one = rearrange(inp_dec_level3_one, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
        inp_dec_level3_two = rearrange(inp_dec_level3_two, "b (h w) c -> b c h w", h=H // 4, w=W // 4).contiguous()
    
        inp_dec_level3_one = self.reduce_chan_level3(inp_dec_level3_one)
        inp_dec_level3_two = self.reduce_chan_level3(inp_dec_level3_two)
    
        inp_dec_level3_one = rearrange(inp_dec_level3_one, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        inp_dec_level3_two = rearrange(inp_dec_level3_two, "b c h w -> b (h w) c").contiguous()  # b, hw//16, 4c
        out_dec_level3_one = inp_dec_level3_one
        out_dec_level3_two = inp_dec_level3_two
   
        for layer in self.decoder_level3:
            out_dec_level3_one, out_dec_level3_two = layer(out_dec_level3_one, out_dec_level3_two, [H // 4, W // 4])

        inp_dec_level2_one = self.up3_2(out_dec_level3_one, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2_two = self.up3_2(out_dec_level3_two, H // 4, W // 4)  # b, hw//4, 2c   
        inp_dec_level2_one = torch.cat([inp_dec_level2_one, out_enc_level2_one], 2)
        inp_dec_level2_two = torch.cat([inp_dec_level2_two, out_enc_level2_two], 2)

        inp_dec_level2_one = rearrange(inp_dec_level2_one, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2_two = rearrange(inp_dec_level2_two, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()  

        inp_dec_level2_one = self.reduce_chan_level2(inp_dec_level2_one)
        inp_dec_level2_two = self.reduce_chan_level2(inp_dec_level2_two)

        inp_dec_level2_one = rearrange(inp_dec_level2_one, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
        inp_dec_level2_two = rearrange(inp_dec_level2_two, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c
     
        out_dec_level2_one = inp_dec_level2_one
        out_dec_level2_two = inp_dec_level2_two


        for layer in self.decoder_level2:
            out_dec_level2_one, out_dec_level2_two = layer(out_dec_level2_one, out_dec_level2_two, [H // 2, W // 2])

        inp_dec_level1_one = self.up2_1(out_dec_level2_one, H // 2, W // 2)  # b, hw, c
        inp_dec_level1_two = self.up2_1(out_dec_level2_two, H // 2, W // 2)  # b, hw, c   
        inp_dec_level1_one = torch.cat([inp_dec_level1_one, out_enc_level1_one], 2)
        inp_dec_level1_two = torch.cat([inp_dec_level1_two, out_enc_level1_two], 2)

        out_dec_level1_one = inp_dec_level1_one
        out_dec_level1_two = inp_dec_level1_two
        for layer in self.decoder_level1:
            out_dec_level1_one, out_dec_level1_two = layer(out_dec_level1_one, out_dec_level1_two, [H, W])

        for layer in self.refinement:
            out_dec_level1_one, out_dec_level1_two = layer(out_dec_level1_one, out_dec_level1_two, [H, W])
            
        out_dec_level1_one = rearrange(out_dec_level1_one, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        out_dec_level1_two = rearrange(out_dec_level1_two, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1_one = out_dec_level1_one + self.skip_conv(out_dec_level1_one)
            out_dec_level1_one = self.output(out_dec_level1_one)
            out_dec_level1_two = out_dec_level1_two + self.skip_conv(out_dec_level1_two)
            out_dec_level1_two = self.output(out_dec_level1_two)
        ##########################
        else:
            out_dec_level1_one = self.output(out_dec_level1_one)  + inp_img_VI + inp_img_IR
            out_dec_level1_two = self.output(out_dec_level1_two) + inp_img_VI + inp_img_IR
        # vutils.save_image(out_dec_level1_one, 'reconstructed_image.png', normalize=True)
        # vutils.save_image(out_dec_level1_two, 'reconstructed_image_new.png', normalize=True)
        return out_dec_level1_one, out_dec_level1_two





if __name__ == "__main__":
    batch_size = 2
    imageA = torch.randn(batch_size, 3, 16, 16).to('cuda:1')
    imageB = torch.randn(batch_size, 3, 16, 16).to('cuda:1')

    model = WaveMamba(        
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        mlp_ratio=2.,
        bias=False,
        dual_pixel_task=False).to('cuda:1')
    # model = DistributedDataParallel(model, find_unused_parameters=True)
    # output1,output2 = model(imageB, imageA)
    # print(output.shape)

    # 假设这是你的损失函数
    print(model)  # 检查所有模块是否按预期初始化
    def loss_fn(output, target):
        return torch.nn.functional.mse_loss(output, target)

    # 模拟输入数据
    input_data = torch.randn(1, 3, 16, 16).to('cuda:1')
    target = torch.randn(2, 3, 16, 16).to('cuda:1')
    imageA = torch.randn(batch_size, 3, 16, 16).to('cuda:1')
    imageB = torch.randn(batch_size, 3, 16, 16).to('cuda:1')
 

    # 前向传播
    output1,output2 = model(imageB, imageA)
    loss1 = loss_fn(output1, target)
    loss2 = loss_fn(output2, target)
    # 反向传播
    loss = loss1+loss2
    loss.backward()

    # 检查哪些参数没有参与到损失计算中
    unused_parameters = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_parameters.append(name)

    print("未使用的参数:", unused_parameters)