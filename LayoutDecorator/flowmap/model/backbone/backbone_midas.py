from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..projection import earlier, later, sample_image_grid
from .backbone import Backbone, BackboneOutput


def make_net(dims):
    def init_weights_normal(m):
        if type(m) == nn.Linear:
            if hasattr(m, "weight"):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, nonlinearity="relu", mode="fan_in"
                )

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    net = nn.Sequential(*layers[:-1])
    net.apply(init_weights_normal)
    return net


@dataclass
class BackboneMidasCfg:
    name: Literal["midas"]
    pretrained: bool
    weight_sensitivity: float | None
    mapping: Literal["original", "exp"]
    model: Literal["DPT_Large", "MiDaS_small"]


class BackboneMidas(Backbone[BackboneMidasCfg]):
    def __init__(
        self,
        cfg: BackboneMidasCfg,
        num_frames: int | None,
        image_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__(cfg, num_frames=num_frames, image_shape=image_shape)
        self.midas = torch.hub.load(
            "intel-isl/MiDaS",
            cfg.model,
            pretrained=cfg.pretrained,
        )
        self.midas_out = self.midas.scratch.output_conv
        self.midas.scratch.output_conv = nn.Identity()

        # If a weight sensitivity is specified, don't learn weights.
        if cfg.weight_sensitivity is None:
            weight_channels = {
                "DPT_Large": 256,
                "MiDaS_small": 64,
            }[cfg.model]
            self.corr_weighter_perpoint = make_net([weight_channels * 2, 128, 64, 1])
        else:
            weights = torch.full((num_frames - 1, *image_shape), 0, dtype=torch.float32)
            self.weights = nn.Parameter(weights)

        if cfg.mapping == "exp":
            self.midas_out = nn.Sequential(*self.midas_out[:-2])

    def forward(self, batch: Batch, flows: Flows) -> BackboneOutput:
        device = batch.videos.device
        b, f, _, h, w = batch.videos.shape

        videos = rearrange(batch.videos, "b f c h w -> (b f) c h w")
        features = self.midas(videos)

        # This matches Cameron's original implementation.
        match self.cfg.mapping:
            case "original":
                depths = 1e3 / (self.midas_out(features) + 0.1)
            case "exp":
                depths = (self.midas_out(features) / 1000).exp() + 0.01

        features = F.interpolate(features, (h, w), mode="bilinear") / 20

        depths = rearrange(depths, "(b f) () h w -> b f h w", b=b, f=f)
        features = rearrange(features, "(b f) c h w -> b f c h w", b=b, f=f)

        # Compute correspondence weights.
        if self.cfg.weight_sensitivity is None:
            xy, _ = sample_image_grid((h, w), device)
            backward_weights = self.compute_correspondence_weights(
                self.grid_sample_features(earlier(features), xy + flows.backward),
                later(features),
            )
        else:
            backward_weights = (self.cfg.weight_sensitivity * self.weights).sigmoid()
            backward_weights = backward_weights[None]

        return BackboneOutput(depths, backward_weights)

    def compute_correspondence_weights(
        self,
        features_earlier: Float[Tensor, "batch frame channel height width"],
        features_later: Float[Tensor, "batch frame channel height width"],
    ) -> Float[Tensor, "batch frame height width"]:
        features = torch.cat((features_earlier, features_later), dim=2)
        features = rearrange(features, "b f c h w -> b f h w c")
        weights = self.corr_weighter_perpoint(features).sigmoid().clip(min=1e-4)
        return rearrange(weights, "b f h w () -> b f h w")

    def grid_sample_features(
        self,
        features: Float[Tensor, "batch frame channel height width"],
        grid: Float[Tensor, "batch frame height width xy=2"],
    ) -> Float[Tensor, "batch frame channel height width"]:
        b, f, _, _, _ = features.shape
        samples = F.grid_sample(
            rearrange(features, "b f c h w -> (b f) c h w"),
            rearrange(grid * 2 - 1, "b f h w xy -> (b f) h w xy"),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rearrange(samples, "(b f) c h w -> b f c h w", b=b, f=f)

# from dataclasses import dataclass
# from typing import Literal

# import torch
# import torch.nn.functional as F
# from einops import rearrange
# from jaxtyping import Float
# from torch import Tensor, nn

# from ...dataset.types import Batch
# from ...flow.flow_predictor import Flows
# from ..projection import earlier, later, sample_image_grid
# from .backbone import Backbone, BackboneOutput

# def make_net(dims):
#     def init_weights_normal(m):
#         if type(m) == nn.Linear and hasattr(m, "weight"):
#             nn.init.kaiming_normal_(
#                 m.weight, a=0.0, nonlinearity="relu", mode="fan_in"
#             )
#     layers = []
#     for i in range(len(dims) - 1):
#         layers.append(nn.Linear(dims[i], dims[i + 1]))
#         layers.append(nn.ReLU())
#     net = nn.Sequential(*layers[:-1])
#     net.apply(init_weights_normal)
#     return net

# @dataclass
# class BackboneMidasCfg:
#     name: Literal["midas"]
#     pretrained: bool
#     weight_sensitivity: float | None
#     mapping: Literal["original", "exp"]
#     model: Literal["DPT_Large", "MiDaS_small"]

# # Inspired by CBAM and residual learning, we design a CBAM module as before.
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super().__init__()
#         # Channel Attention
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels, bias=False),
#             nn.Sigmoid()
#         )
#         # Spatial Attention
#         self.spatial = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, h, w = x.size()
#         # Channel Attention
#         avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
#         max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
#         channel_attention = avg_out + max_out
#         x = x * channel_attention
#         # Spatial Attention
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_attention = self.spatial(torch.cat([avg_out, max_out], dim=1))
#         x = x * spatial_attention
#         return x

# # Updated Depth Refinement Module with CBAM and a learnable scaling factor.
# class DepthRefinementModule(nn.Module):
#     def __init__(self, in_channels: int = 1, mid_channels: int = 16):
#         super().__init__()
#         self.conv_expand = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
#         self.cbam = CBAM(mid_channels, reduction=4)
#         self.conv_reduce = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
#         # Introduce a learnable scaling factor initialized to zero.
#         self.alpha = nn.Parameter(torch.zeros(1))
#     def forward(self, x):
#         # x: (batch, 1, h, w)
#         residual = x
#         out = self.conv_expand(x)
#         out = self.cbam(out)
#         out = self.conv_reduce(out)
#         # Multiply residual correction by alpha
#         return residual + self.alpha * out

# class BackboneMidas(Backbone[BackboneMidasCfg]):
#     def __init__(
#         self,
#         cfg: BackboneMidasCfg,
#         num_frames: int | None,
#         image_shape: tuple[int, int] | None,
#     ) -> None:
#         super().__init__(cfg, num_frames=num_frames, image_shape=image_shape)
#         self.midas = torch.hub.load(
#             "intel-isl/MiDaS",
#             cfg.model,
#             pretrained=cfg.pretrained,
#         )
#         self.midas_out = self.midas.scratch.output_conv
#         self.midas.scratch.output_conv = nn.Identity()

#         # Setup correspondence weight network (MLP) using make_net.
#         if cfg.weight_sensitivity is None:
#             weight_channels = {
#                 "DPT_Large": 256,
#                 "MiDaS_small": 64,
#             }[cfg.model]
#             self.corr_weighter_perpoint = make_net([weight_channels * 2, 128, 64, 1])
#         else:
#             weights = torch.full((num_frames - 1, *image_shape), 0, dtype=torch.float32)
#             self.weights = nn.Parameter(weights)

#         if cfg.mapping == "exp":
#             self.midas_out = nn.Sequential(*self.midas_out[:-2])
        
#         # Add our Depth Refinement Module to refine the initial depth prediction.
#         self.depth_refinement = DepthRefinementModule(in_channels=1, mid_channels=16)

#     def forward(self, batch: Batch, flows: Flows) -> BackboneOutput:
#         device = batch.videos.device
#         b, f, _, h, w = batch.videos.shape

#         # Rearrange video frames to shape (b*f, c, h, w)
#         videos = rearrange(batch.videos, "b f c h w -> (b f) c h w")
#         features = self.midas(videos)

#         # Compute initial depth predictions based on mapping type.
#         match self.cfg.mapping:
#             case "original":
#                 depths = 1e3 / (self.midas_out(features) + 0.1)
#             case "exp":
#                 depths = (self.midas_out(features) / 1000).exp() + 0.01

#         features = F.interpolate(features, (h, w), mode="bilinear") / 20

#         depths = rearrange(depths, "(b f) () h w -> b f h w", b=b, f=f)
#         features = rearrange(features, "(b f) c h w -> b f c h w", b=b, f=f)

#         # Apply the Depth Refinement Module.
#         # Expand depth to include a channel dimension: (b, f, 1, h, w)
#         depth_initial = depths.unsqueeze(1)
#         # Merge batch and frame dimensions to process with conv layers.
#         depth_initial_4d = depth_initial.view(b * f, 1, h, w)
#         depth_refined_4d = self.depth_refinement(depth_initial_4d)
#         depth_refined = depth_refined_4d.view(b, f, h, w)

#         # Compute correspondence weights.
#         if self.cfg.weight_sensitivity is None:
#             xy, _ = sample_image_grid((h, w), device)
#             backward_weights = self.compute_correspondence_weights(
#                 self.grid_sample_features(earlier(features), xy + flows.backward),
#                 later(features),
#             )
#         else:
#             backward_weights = (self.cfg.weight_sensitivity * self.weights).sigmoid()
#             backward_weights = backward_weights[None]

#         return BackboneOutput(depth_refined, backward_weights)

#     def compute_correspondence_weights(
#         self,
#         features_earlier: Float[Tensor, "batch frame channel height width"],
#         features_later: Float[Tensor, "batch frame channel height width"],
#     ) -> Float[Tensor, "batch frame height width"]:
#         features = torch.cat((features_earlier, features_later), dim=2)
#         features = rearrange(features, "b f c h w -> b f h w c")
#         weights = self.corr_weighter_perpoint(features).sigmoid().clip(min=1e-4)
#         return rearrange(weights, "b f h w () -> b f h w")

#     def grid_sample_features(
#         self,
#         features: Float[Tensor, "batch frame channel height width"],
#         grid: Float[Tensor, "batch frame height width xy=2"],
#     ) -> Float[Tensor, "batch frame channel height width"]:
#         b, f, _, H, W = features.shape
#         # Clamp grid coordinates to valid pixel ranges
#         grid[..., 0] = grid[..., 0].clamp(0, W - 1)
#         grid[..., 1] = grid[..., 1].clamp(0, H - 1)
#         # Convert pixel coordinates to normalized coordinates [-1, 1]
#         grid_norm = grid.clone()
#         grid_norm[..., 0] = grid[..., 0] / (W - 1) * 2 - 1
#         grid_norm[..., 1] = grid[..., 1] / (H - 1) * 2 - 1
#         samples = F.grid_sample(
#             rearrange(features, "b f c h w -> (b f) c h w"),
#             rearrange(grid_norm, "b f h w xy -> (b f) h w xy"),
#             mode="bilinear",
#             padding_mode="zeros",
#             align_corners=False,
#         )
#         return rearrange(samples, "(b f) c h w -> b f c h w", b=b, f=f)


