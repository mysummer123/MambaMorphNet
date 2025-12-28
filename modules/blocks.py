# modules/blocks.py

# -*- coding: utf-8 -*-
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None
    print("Warning: mamba_ssm is not installed. Mamba blocks will not work.")


class Mamba(nn.Module):
    """
    Standard Mamba block (Linear Complexity Sequence Modeling).
    Kept as the primitive operator.
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(
            dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        B, L, _ = hidden_states.shape
        if self.use_fast_path and mamba_inner_fn is not None:
            xz = self.in_proj.weight @ hidden_states.transpose(1, 2)
            if self.in_proj.bias is not None:
                xz = xz + self.in_proj.bias.unsqueeze(-1)
            out = mamba_inner_fn(
                xz, self.conv1d.weight, self.conv1d.bias, self.x_proj.weight, self.dt_proj.weight,
                self.out_proj.weight, self.out_proj.bias, -torch.exp(self.A_log.float()), None, None,
                self.D.float(), delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
            return out.transpose(1, 2)
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv1d(x.transpose(1, 2)).transpose(1, 2))
        x_dbl = self.x_proj(x)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        y = selective_scan_fn(
            x, dt, -torch.exp(self.A_log.float()), B_ssm.contiguous(), C_ssm.contiguous(), self.D.float(),
            z.contiguous(), self.dt_proj.bias.float(), True
        )
        return self.out_proj(y)


class AdaptiveGate(nn.Module):
    """
    Direction-Aware Gating Mechanism.
    Predicts weights for fusing scanning directions.
    """
    def __init__(self, dim, num_heads=2): # num_heads=2 corresponds to (Row/Col) inside a group
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_heads),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        return self.gate_net(x)


class LeFF(nn.Module):
    """
    Locally Enhanced Feed-Forward (LeFF).
    Now used INSIDE the HGL-SSM groups.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        # Input x: (B, L, C)
        B, L, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        # Reshape to Image for DWConv
        x = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        x = self.conv(x)
        # Flatten back to Sequence
        x = x.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HGLSSM(nn.Module):
    """
    Hybrid Global-Local SSM (HGL-SSM).
    The Core Innovation:
    1. Dual-Stream Scanning (Fwd Group, Bwd Group)
    2. Intra-Group LeFF Refinement (Standard LeFF is applied here)
    3. Global Integration
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        
        # 4 Mamba Scanners
        self.ssm_row_fwd = Mamba(d_model, d_state, d_conv, expand, **kwargs)
        self.ssm_col_fwd = Mamba(d_model, d_state, d_conv, expand, **kwargs)
        self.ssm_row_bwd = Mamba(d_model, d_state, d_conv, expand, **kwargs)
        self.ssm_col_bwd = Mamba(d_model, d_state, d_conv, expand, **kwargs)

        # Adaptive Gating for Group Fusion
        # We need 4 weights (2 for fwd group, 2 for bwd group)
        self.gate_fwd = AdaptiveGate(d_model, num_heads=2)
        self.gate_bwd = AdaptiveGate(d_model, num_heads=2)

        # LeFF Modules for Local Refinement (one per group)
        # Ratio can be smaller here since it's inside the block
        self.leff_fwd = LeFF(d_model, int(d_model * 2)) 
        self.leff_bwd = LeFF(d_model, int(d_model * 2))

        # Final Linear Fusion
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2) # (B, C, H, W) used for gating

        # --- Stage 1: Dual-Stream Scanning ---
        
        # Prepare Sequences
        x_2d = x.view(B, H, W, C)
        
        # Forward Stream (Top-Left Origin)
        x_row_fwd_in = x_2d.permute(0, 2, 1, 3).contiguous().view(B * W, H, C) # Row scan
        x_col_fwd_in = x_2d.permute(0, 1, 2, 3).contiguous().view(B * H, W, C) # Col scan
        
        y_row_fwd = self.ssm_row_fwd(x_row_fwd_in).view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)
        y_col_fwd = self.ssm_col_fwd(x_col_fwd_in).view(B, H, W, C).reshape(B, L, C)

        # Backward Stream (Bottom-Right Origin)
        # Flip input for backward scan
        x_row_bwd_in = x_2d.flip(dims=[1]).permute(0, 2, 1, 3).contiguous().view(B * W, H, C)
        x_col_bwd_in = x_2d.flip(dims=[1]).permute(0, 1, 2, 3).contiguous().view(B * H, W, C)

        # Scan and then Flip back
        y_row_bwd = self.ssm_row_bwd(x_row_bwd_in).view(B, W, H, C).permute(0, 2, 1, 3).flip(dims=[1]).reshape(B, L, C)
        y_col_bwd = self.ssm_col_bwd(x_col_bwd_in).view(B, H, W, C).flip(dims=[1]).reshape(B, L, C)

        # --- Stage 2: Causal-Aware Grouping & LeFF Refinement ---
        
        # Group 1: Forward
        w_fwd = self.gate_fwd(x_img) # (B, 2)
        grp_fwd = w_fwd[:, 0].view(B, 1, 1) * y_row_fwd + w_fwd[:, 1].view(B, 1, 1) * y_col_fwd
        z_fwd = self.leff_fwd(grp_fwd, H, W) # Local Refinement

        # Group 2: Backward
        w_bwd = self.gate_bwd(x_img) # (B, 2)
        grp_bwd = w_bwd[:, 0].view(B, 1, 1) * y_row_bwd + w_bwd[:, 1].view(B, 1, 1) * y_col_bwd
        z_bwd = self.leff_bwd(grp_bwd, H, W) # Local Refinement

        # --- Stage 3: Global Integration ---
        out = self.out_proj(z_fwd + z_bwd)
        
        return out + x # Residual connection within module


class HGLBlock(nn.Module):
    """
    High-level Block wrapping HGL-SSM.
    Replaces the old MambaLeWinBlock.
    Structure: Norm -> HGL-SSM -> Norm -> FFN (Optional, usually removed if LeFF is inside)
    Since LeFF is inside HGL-SSM, we just need a simple projection or just the SSM part.
    Based on the paper design: U-Net stages use HGL-SSM blocks.
    """
    def __init__(self, dim, drop_path=0., norm_layer=nn.LayerNorm, **mamba_kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        self.hgl_ssm = HGLSSM(d_model=dim, **mamba_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        # x: (B, L, C)
        shortcut = x
        x = self.norm(x)
        x = self.hgl_ssm(x, H, W)
        x = shortcut + self.drop_path(x)
        return x

# --- Helper Blocks for U-Net Structure ---

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation is None:
            self.act = None
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Module):
    def __init__(self, num_filter, k=3, s=1, p=1, bias=True, act='prelu', norm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, k, s, p, bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, k, s, p, bias=bias)
        self.norm = norm
        if self.norm is not None:
            self.bn = nn.BatchNorm2d(num_filter) if norm == 'batch' else nn.InstanceNorm2d(num_filter)
        self.act = act
        if self.act is not None:
            self.relu = nn.PReLU() if act == 'prelu' else nn.ReLU(True)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        if self.norm is not None: out = self.bn(out)
        if self.act is not None: out = self.relu(out)
        out = self.conv2(out)
        if self.norm is not None: out = self.bn(out)
        out = torch.add(out, res)
        if self.act is not None: out = self.relu(out)
        return out

# The rest (VDCM, CDM, etc.) can be kept in model_cbct2sct.py or here as preferred.
# Assuming they stay in blocks.py for import convenience, I will include the other 3 modules below 
# just to make sure the file is complete and compatible.

class VolumetricDiscrepancyCompensationModule(nn.Module):
    def __init__(self, n_slices, in_ch=1, base_channels=128):
        super().__init__()
        self.n_slices = n_slices
        self.center_feat = ConvBlock(in_ch, base_channels, 3, 1, 1, activation='prelu')
        self.diff_feat = ConvBlock(in_ch, 64, 3, 1, 1, activation='prelu')
        self.fusion_conv = ConvBlock((n_slices - 1) * 64, base_channels, 3, 1, 1, activation='prelu')
        res_body1 = [ResnetBlock(base_channels, norm=None), ConvBlock(base_channels, 64, 3, 1, 1, activation='prelu')]
        self.res_branch1 = nn.Sequential(*res_body1)
        res_body2 = [ResnetBlock(base_channels, norm=None), ConvBlock(base_channels, 64, 3, 1, 1, activation='prelu')]
        self.res_branch2 = nn.Sequential(*res_body2)

    def forward(self, center_slice, neighbor_slices):
        seq = list(neighbor_slices)
        seq.insert(self.n_slices // 2, center_slice)
        diffs = [seq[i] - seq[i + 1] for i in range(self.n_slices - 1)]
        diff_stack = torch.stack(diffs, dim=1)
        B, N, C, H, W = diff_stack.size()
        center_f = self.center_feat(center_slice)
        diff_f = self.diff_feat(diff_stack.view(-1, C, H, W))
        down_diff = F.avg_pool2d(diff_f, 2).view(B, N, -1, H // 2, W // 2)
        stack_f = torch.cat([down_diff[:, j] for j in range(N)], dim=1)
        stack_f = self.fusion_conv(stack_f)
        up2 = F.interpolate(stack_f, scale_factor=2, mode='bilinear', align_corners=True)
        up1 = F.interpolate(self.res_branch1(stack_f), scale_factor=2, mode='bilinear', align_corners=True)
        comp = 0.5 * center_f + 0.5 * up2
        comp = self.res_branch2(comp)
        return 0.5 * comp + 0.5 * up1


class ContextualDisambiguationModule(nn.Module):
    def __init__(self, n_slices, channels=64):
        super().__init__()
        self.wide_conv = ConvBlock(n_slices * channels, channels, 3, 1, 1, activation='prelu')
        self.narrow_conv = ConvBlock((n_slices - 2) * channels, channels, 3, 1, 1, activation='prelu')
        self.proc_conv1 = ConvBlock(channels, channels, 3, 1, 1, activation='prelu')
        self.proc_conv2 = ConvBlock(channels, channels, 3, 1, 1, activation='prelu')
        self.proc_conv3 = ConvBlock(channels, channels, 3, 1, 1, activation='prelu')
        self.attention_gen = ConvBlock(channels, n_slices * channels, 3, 1, 1, activation='prelu')
        self.w1, self.w2 = nn.Parameter(torch.randn(1)), nn.Parameter(torch.randn(1))

    def forward(self, slice_feat_list):
        full_feat = torch.cat(slice_feat_list, dim=1)
        narrow_feat_list = slice_feat_list[1:-1]
        narrow_cat_feat = torch.cat(narrow_feat_list, dim=1)
        wide_f_base = self.wide_conv(full_feat)
        narrow_f_base = self.narrow_conv(narrow_cat_feat)
        wide_f = self.proc_conv1(wide_f_base)
        narrow_f = self.proc_conv1(narrow_f_base)
        diff1, diff2 = wide_f - narrow_f, narrow_f - wide_f
        up1 = F.interpolate(self.proc_conv3(F.avg_pool2d(wide_f, 2)), scale_factor=2, mode='bilinear',align_corners=True)
        up2 = F.interpolate(self.proc_conv3(F.avg_pool2d(narrow_f, 2)), scale_factor=2, mode='bilinear',align_corners=True)
        enh1, enh2 = self.proc_conv2(wide_f), self.proc_conv2(narrow_f)
        att1 = torch.sigmoid(self.attention_gen(diff1 + enh1 + up1))
        att2 = torch.sigmoid(self.attention_gen(diff2 + enh2 + up2))
        return (self.w1 * att1 + self.w2 * att2) * full_feat + full_feat


class VolumetricSceneReconstructionModule(nn.Module):
    def __init__(self, channels=64, n_slices=5):
        super().__init__()
        self.center_idx = n_slices // 2
        self.query_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.key_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fusion_conv = nn.Conv2d(n_slices * channels, channels, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, slice_features):
        B, N, C, H, W = slice_features.size()
        center_feat_blueprint = slice_features[:, self.center_idx].clone()

        q = self.query_conv(center_feat_blueprint)
        k = self.key_conv(slice_features.view(-1, C, H, W)).view(B, N, C, H, W)

        consistency_scores = torch.sum(k * q.unsqueeze(1), dim=2, keepdim=True)
        attention_map = torch.sigmoid(consistency_scores)

        attention_map = attention_map.repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        weighted_feats = slice_features.reshape(B, -1, H, W) * attention_map

        return self.lrelu(self.fusion_conv(weighted_feats))


class OrthogonalDecouplingAttentionModule(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels % self.groups == 0
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1_fusion = ConvBlock(channels // self.groups, channels // self.groups, 1, 1, 0)
        self.conv3x3_local = ConvBlock(channels // self.groups, channels // self.groups, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)

        x_h_pooled = self.pool_h(group_x)
        x_w_pooled = self.pool_w(group_x).permute(0, 1, 3, 2)

        hw_fused = self.conv1x1_fusion(torch.cat([x_h_pooled, x_w_pooled], dim=2))
        x_h_gate, x_w_gate = torch.split(hw_fused, [h, w], dim=2)

        x_gated = self.gn(group_x * torch.sigmoid(x_h_gate) * torch.sigmoid(x_w_gate.permute(0, 1, 3, 2)))
        x_local = self.conv3x3_local(group_x)

        gated_stats = F.softmax(F.adaptive_avg_pool2d(x_gated, (1, 1)).view(b * self.groups, 1, -1), dim=-1)
        local_flat = x_local.view(b * self.groups, c // self.groups, -1)

        local_stats = F.softmax(F.adaptive_avg_pool2d(x_local, (1, 1)).view(b * self.groups, 1, -1), dim=-1)
        gated_flat = x_gated.view(b * self.groups, c // self.groups, -1)

        weights = (torch.bmm(gated_stats, local_flat) + torch.bmm(local_stats, gated_flat)).view(b * self.groups, 1, h,
                                                                                                 w)

        return (group_x * torch.sigmoid(weights)).reshape(b, c, h, w)