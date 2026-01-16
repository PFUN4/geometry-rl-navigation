import torch
import torch.nn as nn
import torch.nn.functional as F


class RotConv2d(nn.Module):
    """
    Rotation-equivariant convolution (discrete 0/90/180/270 deg).
    Applies K rotated kernels and max-pools across orientations.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, K=4, padding=1):
        super().__init__()
        self.K = K
        self.padding = padding
        self.base_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

    @staticmethod
    def rotate_kernel(w, k: int):
        return torch.rot90(w, k, dims=[-2, -1])

    def forward(self, x):
        feats = []
        for k in range(self.K):
            w = self.rotate_kernel(self.base_weight, k)
            y = F.conv2d(x, w, padding=self.padding)
            feats.append(y)
        return torch.max(torch.stack(feats, dim=0), dim=0).values


class EquivariantCNN(nn.Module):
    """
    Lightweight SE(2)-equivariant-ish encoder for 84x84 RGB.

    Input:  (B, 3, 84, 84)
    Output: (B, 256)
    """
    def __init__(self):
        super().__init__()
        self.layer1 = RotConv2d(3, 32, K=4)
        self.layer2 = RotConv2d(32, 64, K=4)
        self.layer3 = RotConv2d(64, 128, K=4)
        self.pool = nn.MaxPool2d(2, 2)

        # 84 -> 42 -> 21 -> 10
        self.fc = nn.Linear(128 * 10 * 10, 256)

    def forward(self, x):
        x = F.relu(self.layer1(x)); x = self.pool(x)
        x = F.relu(self.layer2(x)); x = self.pool(x)
        x = F.relu(self.layer3(x)); x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)
