# train.py

# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import math
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.data_cbct2sct_nifti import get_training_set_nifti, get_eval_set_nifti
from modules.model_cbct2sct import MambaMorphNet 

warnings.filterwarnings("ignore", category=UserWarning)


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cur = self.last_epoch
        if cur < self.warmup_epochs:
            return [base_lr * (cur + 1) / self.warmup_epochs for base_lr in self.base_lrs]

        eff = cur - self.warmup_epochs
        total = self.max_epochs - self.warmup_epochs
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * eff / total)) / 2
                for base_lr in self.base_lrs]


# Utility Functions
def set_seed(seed: int, cuda: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def psnr_torch(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def print_network(net: nn.Module):
    n_params = sum(p.numel() for p in net.parameters())
    print(net)
    print(f"Total parameters: {n_params / 1e6:.3f} M")


def save_ckpt(model: nn.Module, save_dir: str, name: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, path)
    print(f"[Checkpoint] Saved to: {path}")


def _pad_to_cover(h, w, patch, stride):
    def _cover_len(L):
        if L <= patch: return patch
        n = math.ceil((L - patch) / stride) + 1
        return (n - 1) * stride + patch

    return _cover_len(h), _cover_len(w)


def _do_pad(t: torch.Tensor, size_hw, mode="reflect"):
    _, _, H, W = t.shape
    Hp, Wp = size_hw
    ph, pw = Hp - H, Wp - W
    if ph == 0 and pw == 0:
        return t
    return F.pad(t, (0, pw, 0, ph), mode=mode)


@torch.no_grad()
def infer_sliding_window(model, inp, neighbors, patch, stride, pad_mode, amp):
    device = next(model.parameters()).device
    B, C, H, W = inp.shape
    Hp, Wp = _pad_to_cover(H, W, patch, stride)
    inp_p = _do_pad(inp, (Hp, Wp), pad_mode).to(device)
    nbs_p = [_do_pad(t, (Hp, Wp), pad_mode).to(device) for t in neighbors]

    acc = torch.zeros((B, 1, Hp, Wp), device=device)
    wgt = torch.zeros((B, 1, Hp, Wp), device=device)

    ys, xs = range(0, Hp - patch + 1, stride), range(0, Wp - patch + 1, stride)
    with autocast(enabled=amp):
        for y in ys:
            for x in xs:
                inp_patch = inp_p[:, :, y:y + patch, x:x + patch]
                nbs_patch = [t[:, :, y:y + patch, x:x + patch] for t in nbs_p]
                pred_patch = model(inp_patch, nbs_patch)
                acc[:, :, y:y + patch, x:x + patch] += pred_patch
                wgt[:, :, y:y + patch, x:x + patch] += 1.0

    pred_full = acc / torch.clamp_min(wgt, 1.0)
    return pred_full[:, :, :H, :W]


class PerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss to preserve textural details.
    Uses ImageNet-pretrained VGG19 features.
    """
    def __init__(self):
        super().__init__()
        # Load VGG19 features
        vgg = models.vgg19(pretrained=True).features
        # We use the first few layers (e.g., up to ReLU_5_4 is common, here we use a lighter subset for efficiency)
        # Using layers up to 35 covers deep enough features without excessive computation
        self.slice = nn.Sequential()
        for x in range(35):
            self.slice.add_module(str(x), vgg[x])
        
        # Freeze parameters
        for p in self.slice.parameters():
            p.requires_grad = False
        self.slice.eval()

        # ImageNet normalization mean/std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # x, y are (B, 1, H, W) in [0, 1] usually. VGG needs 3 channels.
        # Repeat channels: 1 -> 3
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        
        # Normalize
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        x_feat = self.slice(x)
        y_feat = self.slice(y)
        
        return F.l1_loss(x_feat, y_feat)


class GradientLoss(nn.Module):
    """
    Gradient Consistency Loss to enforce edge sharpness.
    Computes differences in x and y gradients.
    """
    def __init__(self):
        super().__init__()
        # Sobel-like kernel or simple difference
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, pred, gt):
        # pred, gt: (B, 1, H, W)
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        
        gt_grad_x = F.conv2d(gt, self.kernel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, padding=1)
        
        loss = F.l1_loss(pred_grad_x, gt_grad_x) + F.l1_loss(pred_grad_y, gt_grad_y)
        return loss


# Main Function
def main():
    cfg = {
        # Data Paths
        "train_root": "./dataset/Task2/brain-0",
        "val_root": "./dataset/Task2/brain-0",

        # Data Hyperparameters
        "n_slices": 5,
        "patch": 192,
        "augment": True,
        "cbct_windows": [
            (0, 3000),
            (-500, 2000),
        ],
        "ct_windows": [
            (-150, 250),
            (0, 80),
            (300, 1500),
        ],

        # Model Core Hyperparameters
        "embed_dim": 64,
        "num_blocks": 6,
        "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,

        # Data Split
        "val_ratio": 0.2, "split_seed": 123, "val_patch_size": 192,

        # Training Hyperparameters
        "epochs": 300, "batch_size": 8, "virtual_batch_size": 32, "val_batch_size": 8,
        "lr": 1e-4, "warmup_epochs": 5, "weight_decay": 0.05, "eta_min": 1e-6,
        "threads": 8, "seed": 123, "gpus": 1, "start_epoch": 1,

        "loss_weight_rec": 10.0,   # Pixel-wise L1
        "loss_weight_perc": 0.1,   # Perceptual VGG
        "loss_weight_grad": 1.0,   # Gradient Consistency

        # Validation Sliding Window
        "val_sliding": True, "val_patch": 192, "val_stride": 96, "val_pad_mode": "reflect",

        # Log/Checkpoint
        "snapshots": 10,
        "save_dir": "results_mambamorph_net",
        "log_dir": "results_mambamorph_net/runs",

        # Early Stopping
        "early_stop_patience": 10, "early_stop_min_delta": 0.05,
    }

    print("========= MambaMorphNet Model Config =========")
    [print(f"{k}: {v}") for k, v in cfg.items()]
    print("================================================")

    cuda_ok = torch.cuda.is_available() and cfg["gpus"] >= 1
    set_seed(cfg["seed"], cuda_ok)
    cudnn.benchmark = True

    # Data Loading
    print("===> Loading datasets with Multi-Window strategy")
    train_set = get_training_set_nifti(
        data_dir=cfg["train_root"],
        cbct_windows=cfg["cbct_windows"],
        ct_windows=cfg["ct_windows"],
        nFrames=cfg["n_slices"],
        patch_size=cfg["patch"],
        augment=cfg["augment"]
    )
    val_set = get_eval_set_nifti(
        data_dir=cfg["val_root"],
        cbct_windows=cfg["cbct_windows"],
        ct_windows=cfg["ct_windows"],
        nFrames=cfg["n_slices"],
        patch_size=cfg["val_patch_size"]
    )

    train_loader = DataLoader(
        dataset=train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["threads"], pin_memory=cuda_ok,
        persistent_workers=True if cfg["threads"] > 0 else False
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=cfg["val_batch_size"], shuffle=False,
        num_workers=cfg["threads"], pin_memory=cuda_ok,
        persistent_workers=True if cfg["threads"] > 0 else False
    )

    # Model Initialization
    print("===> Building MambaMorphNet")
    n_input_channels = len(cfg["cbct_windows"])
    print(f"Model input channels: {n_input_channels}")

    model = MambaMorphNet(
        n_slices=cfg["n_slices"],
        in_ch=n_input_channels,
        embed_dim=cfg["embed_dim"],
        num_blocks=cfg["num_blocks"],
        mamba_d_state=cfg["mamba_d_state"],
        mamba_d_conv=cfg["mamba_d_conv"],
        mamba_expand=cfg["mamba_expand"]
    )
    print_network(model)
    if cuda_ok:
        if cfg["gpus"] > 1: model = nn.DataParallel(model, device_ids=list(range(cfg["gpus"])))
        model = model.cuda()

    train_losses, val_psnrs = [], []
    
    criterion_Rec = nn.L1Loss().cuda() if cuda_ok else nn.L1Loss()
    criterion_Perc = PerceptualLoss().cuda() if cuda_ok else PerceptualLoss()
    criterion_Grad = GradientLoss().cuda() if cuda_ok else GradientLoss()

    # Optimizer & LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = WarmupCosineAnnealingLR(optimizer, cfg["warmup_epochs"], cfg["epochs"], cfg["eta_min"])
    
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(cfg["log_dir"])
    except ImportError:
        writer = None
        print("TensorboardX not found, skipping logs.")
        
    scaler = GradScaler()

    best_psnr, best_epoch, no_improve_count = -1.0, -1, 0
    accumulation_steps = max(1, cfg['virtual_batch_size'] // cfg['batch_size'])
    print(f"Gradient Accumulation enabled. Virtual batch size: {cfg['virtual_batch_size']}, "
          f"Physical batch size: {cfg['batch_size']}, Accumulation steps: {accumulation_steps}")

    # Training Loop
    for epoch in range(cfg["start_epoch"], cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg['epochs']}]", ncols=120)
        optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            gt, inp, neighbor, _ = batch

            gt_primary = gt[:, 0:1, :, :].cuda(non_blocking=True)
            inp = inp.cuda(non_blocking=True)
            neighbor = [t.cuda(non_blocking=True) for t in neighbor]

            with autocast():
                pred = model(inp, neighbor)
                
                loss_rec = criterion_Rec(pred, gt_primary)
                loss_perc = criterion_Perc(pred, gt_primary)
                loss_grad = criterion_Grad(pred, gt_primary)
                
                loss = (cfg["loss_weight_rec"] * loss_rec + 
                        cfg["loss_weight_perc"] * loss_perc + 
                        cfg["loss_weight_grad"] * loss_grad)

                loss /= accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accumulation_steps
            pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"===> Epoch {epoch} Complete: Avg Loss {avg_loss:.6f}")
        if writer: writer.add_scalar("train/loss", avg_loss, epoch)

        # Validation
        model.eval()
        psnr_sum = 0.0
        vbar = tqdm(val_loader, desc=f"Validating (epoch {epoch})", ncols=120)

        with torch.no_grad():
            for i, batch in enumerate(vbar, 1):
                gt, inp, neighbor, _ = batch
                gt_primary = gt[:, 0:1, :, :].cuda(non_blocking=True)
                inp = inp.cuda(non_blocking=True)
                neighbor = [t.cuda(non_blocking=True) for t in neighbor]

                if cfg.get("val_sliding"):
                    pred = infer_sliding_window(model, inp, neighbor, cfg["val_patch"], cfg["val_stride"],
                                                cfg["val_pad_mode"], True)
                else:
                    with autocast():
                        pred = model(inp, neighbor)

                psnr_sum += psnr_torch(pred.clamp(0, 1), gt_primary.clamp(0, 1), max_val=1.0)
                vbar.set_postfix(psnr=f"{psnr_sum / i:.3f} dB")

        avg_psnr = psnr_sum / len(val_loader)
        val_psnrs.append(avg_psnr)
        print(f"[VAL] Epoch {epoch}: PSNR {avg_psnr:.3f} dB")
        if writer: writer.add_scalar("val/psnr", avg_psnr, epoch)

        # LR Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"[LR] Updated to => {current_lr:.2e}")
        if writer: writer.add_scalar("train/lr", current_lr, epoch)

        # Checkpoint & Early Stopping
        if avg_psnr > best_psnr + float(cfg["early_stop_min_delta"]):
            best_psnr, best_epoch, no_improve_count = avg_psnr, epoch, 0
            save_ckpt(model, cfg["save_dir"], f"best_model_epoch{epoch}_psnr{avg_psnr:.2f}.pth")
        else:
            no_improve_count += 1
            print(f"[EarlyStop] No improvement for {no_improve_count}/{cfg['early_stop_patience']} epochs "
                  f"(best={best_psnr:.3f} dB @ {best_epoch})")
            if no_improve_count >= cfg["early_stop_patience"]:
                print(f"[EarlyStop] Stop at epoch {epoch}. Best epoch = {best_epoch}, Best PSNR = {best_psnr:.3f} dB")
                break

        if (epoch % cfg["snapshots"]) == 0:
            save_ckpt(model, cfg["save_dir"], f"ckpt_epoch{epoch}.pth")

    print(f"Training done. Best epoch = {best_epoch}, Best PSNR = {best_psnr:.3f} dB")
    if 'writer' in locals() and writer:
        writer.close()

    # Plot Training Curves
    print("===> Plotting training curves...")
    os.makedirs(cfg["save_dir"], exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Training Metrics', fontsize=16)

    epochs_range = range(cfg["start_epoch"], cfg["start_epoch"] + len(train_losses))
    ax1.plot(epochs_range, train_losses, label='Train Loss')
    ax1.set(xlabel='Epoch', ylabel='Average Loss', title='Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, val_psnrs, label='Validation PSNR')
    ax2.set(xlabel='Epoch', ylabel='Average PSNR (dB)', title='Validation PSNR')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(cfg["save_dir"], "training_curves.png")
    plt.savefig(save_path)
    print(f"Training curves saved to: {save_path}")


if __name__ == "__main__":
    main()