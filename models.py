"""DCGAN generator and discriminator for 64x64 RGB face images."""
import torch
from torch import nn


def weights_init(module):
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim=128, base_channels=64, image_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # [B, latent_dim, 1, 1] -> [B, 512, 4, 4]
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            # 4 -> 8
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            # 8 -> 16
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # 16 -> 32
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            # 32 -> 64
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base_channels=64, image_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4 -> 1
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image):
        return self.net(image).view(-1)


def build_models(latent_dim=128, base_channels=64, device="cpu"):
    generator = Generator(latent_dim, base_channels)
    discriminator = Discriminator(base_channels)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator.to(device), discriminator.to(device)
