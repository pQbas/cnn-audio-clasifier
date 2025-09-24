import torch
import torch.nn as nn
from collections import OrderedDict


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1):
        super(ResNetBlock, self).__init__()

        self.transform_map = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(
                c_out,
                c_out,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
        )

        self.ident_map = (
            nn.Identity()
            if c_in == c_out
            else nn.Sequential(
                nn.Conv2d(
                    c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False
                ),
                nn.BatchNorm2d(c_out),
            )
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.transform_map(x) + self.ident_map(x))


class ResNet(nn.Module):
    def __init__(self, categories=50):
        super(ResNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.backbone = nn.Sequential(
            *[
                *[ResNetBlock(64, 64) for _ in range(3)],
                ResNetBlock(64, 128, stride=2),
                *[ResNetBlock(128, 128) for _ in range(3)],
                ResNetBlock(128, 256, stride=2),
                *[ResNetBlock(256, 256) for _ in range(3)],
                ResNetBlock(256, 512, stride=2),
                *[ResNetBlock(512, 512) for _ in range(2)],
            ]
        )

        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, categories)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.backbone(x)
        x = self.decoder(x)
        return x

def get_model():
    model = ResNet()
    return model

