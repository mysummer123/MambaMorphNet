# modules/model_cbct2sct.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.blocks import (
    ConvBlock, ResnetBlock, OrthogonalDecouplingAttentionModule,
    VolumetricDiscrepancyCompensationModule, ContextualDisambiguationModule,
    VolumetricSceneReconstructionModule, HGLBlock
)

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class HybridGlobalLocalDecoder(nn.Module):
    """
    The 4-Stage U-Shaped Decoder described in the paper.
    Replaces the simple cascaded blocks.
    Structure:
    - Encoder Path (Downsampling): HGLBlock -> Downsample
    - Bottleneck
    - Decoder Path (Upsampling): Upsample -> Concat(Skip) -> HGLBlock
    """
    def __init__(self, embed_dim=64, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        
        # --- Encoder Path ---
        # Stage 1: 64
        self.enc1 = HGLBlock(embed_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.down1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=4, stride=2, padding=1)
        
        # Stage 2: 128
        self.enc2 = HGLBlock(embed_dim * 2, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.down2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=4, stride=2, padding=1)
        
        # Stage 3: 256
        self.enc3 = HGLBlock(embed_dim * 4, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        self.down3 = nn.Conv2d(embed_dim * 4, embed_dim * 8, kernel_size=4, stride=2, padding=1)
        
        # --- Bottleneck ---
        # Stage 4: 512
        self.bottleneck = HGLBlock(embed_dim * 8, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        
        # --- Decoder Path ---
        # Up 1 (512 -> 256)
        self.up1 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2)
        self.reduce1 = nn.Conv2d(embed_dim * 8, embed_dim * 4, kernel_size=1) # Reduce after concat
        self.dec1 = HGLBlock(embed_dim * 4, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        
        # Up 2 (256 -> 128)
        self.up2 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, kernel_size=2, stride=2)
        self.reduce2 = nn.Conv2d(embed_dim * 4, embed_dim * 2, kernel_size=1)
        self.dec2 = HGLBlock(embed_dim * 2, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
        
        # Up 3 (128 -> 64)
        self.up3 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=2, stride=2)
        self.reduce3 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        self.dec3 = HGLBlock(embed_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)

    def forward(self, x, H, W):
        # Note: We need to update H, W as we downsample
        
        # Encoder 1
        x1 = self.enc1(x.flatten(2).transpose(1, 2), H, W).transpose(1, 2).view(x.shape[0], -1, H, W)
        d1 = self.down1(x1)
        
        # Encoder 2
        H2, W2 = H // 2, W // 2
        x2 = self.enc2(d1.flatten(2).transpose(1, 2), H2, W2).transpose(1, 2).view(d1.shape[0], -1, H2, W2)
        d2 = self.down2(x2)
        
        # Encoder 3
        H3, W3 = H2 // 2, W2 // 2
        x3 = self.enc3(d2.flatten(2).transpose(1, 2), H3, W3).transpose(1, 2).view(d2.shape[0], -1, H3, W3)
        d3 = self.down3(x3)
        
        # Bottleneck
        H4, W4 = H3 // 2, W3 // 2
        b = self.bottleneck(d3.flatten(2).transpose(1, 2), H4, W4).transpose(1, 2).view(d3.shape[0], -1, H4, W4)
        
        # Decoder 1
        u1 = self.up1(b)
        c1 = torch.cat([u1, x3], dim=1) # Skip Connection
        c1 = self.reduce1(c1)
        dec1_out = self.dec1(c1.flatten(2).transpose(1, 2), H3, W3).transpose(1, 2).view(c1.shape[0], -1, H3, W3)
        
        # Decoder 2
        u2 = self.up2(dec1_out)
        c2 = torch.cat([u2, x2], dim=1)
        c2 = self.reduce2(c2)
        dec2_out = self.dec2(c2.flatten(2).transpose(1, 2), H2, W2).transpose(1, 2).view(c2.shape[0], -1, H2, W2)
        
        # Decoder 3
        u3 = self.up3(dec2_out)
        c3 = torch.cat([u3, x1], dim=1)
        c3 = self.reduce3(c3)
        out = self.dec3(c3.flatten(2).transpose(1, 2), H, W).transpose(1, 2).view(c3.shape[0], -1, H, W)
        
        return out


class MambaMorphNet(nn.Module):
    def __init__(self, n_slices=5, in_ch=1, embed_dim=64, num_blocks=4, # num_blocks param is deprecated but kept for compatibility
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        assert n_slices % 2 == 1, "n_slices must be odd"
        self.n_slices = n_slices
        self.center_idx = n_slices // 2
        self.embed_dim = embed_dim

        # Branch A: Focal-Slice Rectification
        self.shallow_feat_extractor = ConvBlock(in_ch, embed_dim, 3, 1, 1, activation='prelu')
        self.odam = OrthogonalDecouplingAttentionModule(embed_dim, 1)
        self.vdcm = VolumetricDiscrepancyCompensationModule(n_slices=n_slices, in_ch=in_ch,
                                                            base_channels=embed_dim * 2)

        # Branch B: Volumetric Scene Contextualization
        self.multi_slice_extractor = ConvBlock(in_ch, embed_dim, 3, 1, 1, activation='prelu')
        self.multi_slice_enhancer = nn.Sequential(*[ResnetBlock(embed_dim, norm=None) for _ in range(5)])
        self.vsrm = VolumetricSceneReconstructionModule(channels=embed_dim, n_slices=n_slices)
        self.cdm = ContextualDisambiguationModule(n_slices=n_slices, channels=embed_dim)
        self.cdm_fus = ConvBlock(n_slices * embed_dim, embed_dim, 1, 1, 0, activation='prelu')

        # Branch C: Hybrid Global-Local Mamba Decoder (U-Shaped)
        self.decoder = HybridGlobalLocalDecoder(
            embed_dim=embed_dim,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand
        )

        # Reconstruction Head
        self.recon_conv = nn.Conv2d(embed_dim, 1, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, center_slice, neighbor_slices):
        B, C, H, W = center_slice.size()

        # Branch A: Focal-Slice Rectification
        shallow_feat = self.odam(self.shallow_feat_extractor(center_slice))
        compensated_feat = self.vdcm(center_slice, neighbor_slices)
        final_skip_feat = shallow_feat + compensated_feat

        # Branch B: Volumetric Scene Contextualization - Feature Prep
        all_slices = list(neighbor_slices)
        all_slices.insert(self.center_idx, center_slice)
        all_slices_tensor = torch.cat(all_slices, dim=1)
        slice_feats = self.multi_slice_extractor(all_slices_tensor.view(B * self.n_slices, C, H, W))
        enhanced_feats = self.multi_slice_enhancer(slice_feats)
        slice_feats = enhanced_feats.view(B, self.n_slices, self.embed_dim, H, W)

        # Branch B: Parallel Fusion
        fused_feat_vsrm = self.vsrm(slice_feats)
        feat_list = [slice_feats[:, i] for i in range(self.n_slices)]
        fused_feat_cdm = self.cdm(feat_list)
        fused_feat_cdm = self.cdm_fus(fused_feat_cdm)

        # Backbone Input Aggregation
        backbone_input = fused_feat_vsrm + fused_feat_cdm
        
        # Branch C: Hybrid Global-Local Mamba Decoder
        deep_features = self.decoder(backbone_input, H, W)

        # Final Fusion and Reconstruction
        final_features = deep_features + final_skip_feat
        output = self.recon_conv(final_features)

        return output


# Quick self-test
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Test config
    B, C, H, W, n_slices = 2, 1, 192, 192, 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy input
    center_s = torch.randn(B, C, H, W).to(device)
    neighbor_s = [torch.randn(B, C, H, W).to(device) for _ in range(n_slices - 1)]

    # Initialize model
    net = MambaMorphNet(
        n_slices=n_slices, in_ch=C, embed_dim=64,
        mamba_d_state=16, mamba_expand=2
    ).to(device)

    # Test forward pass
    print("--- Testing MambaMorphNet (U-Shaped Decoder) ---")
    with torch.no_grad():
        y = net(center_s, neighbor_s)

    # Validate output shape
    print(f"Input center slice: {center_s.shape}, neighbors: {len(neighbor_s)}x{neighbor_s[0].shape}")
    print(f"Output sCT slice: {y.shape}")
    assert y.shape == center_s.shape, f"Output shape mismatch: {y.shape} vs {center_s.shape}"

    # Calculate trainable parameters
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

    # Memory usage check
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(device) / 1e6
        print(f"GPU memory allocated: {mem_allocated:.2f} MB")
        torch.cuda.empty_call_stack()