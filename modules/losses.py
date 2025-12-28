import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layers=[3, 8, 17, 26, 35]):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        for param in vgg.parameters():
            param.requires_grad = False
        vgg.eval()

        self.features = vgg
        self.layer_indices = feature_layers

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            gt = gt.repeat(1, 3, 1, 1)

        pred = (pred - self.mean) / self.std
        gt = (gt - self.mean) / self.std

        loss = 0.0
        x = pred
        y = gt

        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)

            if i in self.layer_indices:
                loss += self.criterion(x, y)

            if i >= max(self.layer_indices):
                break

        return loss


class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super(GradientConsistencyLoss, self).__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        if pred.shape[1] == 3:
            pred = pred.mean(dim=1, keepdim=True)
            gt = gt.mean(dim=1, keepdim=True)

        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        gt_grad_x = F.conv2d(gt, self.kernel_x, padding=1)

        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, padding=1)

        loss = self.criterion(pred_grad_x, gt_grad_x) + self.criterion(pred_grad_y, gt_grad_y)
        return loss


class MambaMorphLoss(nn.Module):
    def __init__(self, lambda_pix=10.0, lambda_perc=0.1, lambda_grad=1.0):
        super(MambaMorphLoss, self).__init__()

        self.lambda_pix = lambda_pix
        self.lambda_perc = lambda_perc
        self.lambda_grad = lambda_grad

        self.l1_loss = nn.L1Loss()
        self.perc_loss = VGGPerceptualLoss()
        self.grad_loss = GradientConsistencyLoss()

    def forward(self, pred, gt):
        loss_pix = self.l1_loss(pred, gt)
        loss_perc = self.perc_loss(pred, gt)
        loss_grad = self.grad_loss(pred, gt)

        loss_total = (self.lambda_pix * loss_pix +
                      self.lambda_perc * loss_perc +
                      self.lambda_grad * loss_grad)

        return loss_total, {
            "loss_pix": loss_pix.item(),
            "loss_perc": loss_perc.item(),
            "loss_grad": loss_grad.item()
        }