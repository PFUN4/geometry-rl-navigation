import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from geometry_encoders import EquivariantCNN


class EquivRgbGeomExtractor(BaseFeaturesExtractor):
    """
    SB3 features extractor for Dict obs:
      obs["rgb"]: uint8 (H,W,C) but SB3 VecTransposeImage makes it (C,H,W)
      obs["geom"]: float32 (3,)

    Output: concatenated features = [equivariant_rgb(256), geom_mlp(64)] = 320 dims
    """

    def __init__(self, observation_space, geom_dim: int = 3, geom_latent: int = 64):
        # Determine output features dim
        super().__init__(observation_space, features_dim=256 + geom_latent)

        # Visual encoder
        self.rgb_encoder = EquivariantCNN()  # outputs 256

        # Geometry MLP
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.ReLU(),
            nn.Linear(64, geom_latent),
            nn.ReLU(),
        )

    def forward(self, observations):
        # observations["rgb"] is float tensor already normalised by SB3 preprocessing
        rgb = observations["rgb"]
        if rgb.dtype != th.float32:
            rgb = rgb.float()

        # EquivariantCNN expects (B,3,84,84)
        rgb_feat = self.rgb_encoder(rgb)

        geom = observations["geom"]
        if geom.dtype != th.float32:
            geom = geom.float()
        geom_feat = self.geom_mlp(geom)

        return th.cat([rgb_feat, geom_feat], dim=1)
