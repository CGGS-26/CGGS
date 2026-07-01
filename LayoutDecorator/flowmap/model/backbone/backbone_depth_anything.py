from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from ...dataset.types import Batch
from ...flow.flow_predictor import Flows
from ..projection import sample_image_grid
from .backbone import Backbone, BackboneOutput
import numpy as np

def make_net(dims):
    def init_weights_normal(m):
        if isinstance(m, nn.Linear) and hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    net = nn.Sequential(*layers[:-1])
    net.apply(init_weights_normal)
    return net

@dataclass
class BackboneDepthAnythingCfg:
    name: Literal["depth_anything"]
    pretrained: bool
    weight_sensitivity: float | None
    model: Literal["LiheYoung/depth-anything-large-hf"]

class ResidualRefinementBlock(nn.Module):
    """
    A Residual Refinement Block that learns a residual correction to the initial depth prediction.
    This block uses two convolutional layers with batch normalization and a skip connection.
    """
    def __init__(self, in_channels: int = 1, mid_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Add the residual and apply ReLU
        out = self.relu(out + residual)
        return out

class BackboneDepthAnything(Backbone[BackboneDepthAnythingCfg]):
    def __init__(
        self,
        cfg: BackboneDepthAnythingCfg,
        num_frames: int | None,
        image_shape: tuple[int, int] | None,
    ) -> None:
        super().__init__(cfg, num_frames=num_frames, image_shape=image_shape)
        
        # Load the image processor and depth estimation model from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
        
        # Set the model to training mode to allow gradient updates
        self.depth_model.train()
        
        # Define the Residual Refinement Block to refine the depth prediction
        self.refinement_block = ResidualRefinementBlock(in_channels=1, mid_channels=32)
        
        # Optional: Define a depth prediction head for further processing (not used in this implementation)
        feature_channels = 256
        self.depth_head = nn.Sequential(
            nn.Conv2d(feature_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )
        
        # Setup correspondence weight computation network or learnable weights
        if cfg.weight_sensitivity is None:
            self.corr_weighter_perpoint = make_net([feature_channels * 2, 128, 64, 1])
        else:
            weights = torch.full((num_frames - 1, *image_shape), 0, dtype=torch.float32)
            self.weights = nn.Parameter(weights)
    
    def forward(self, batch: Batch, flows: Flows) -> BackboneOutput:
        device = batch.videos.device
        b, f, c, h, w = batch.videos.shape
        
        # Reshape video frames to (b*f, c, h, w)
        frames = batch.videos.view(b * f, c, h, w).clamp(0, 1)
        
        # Normalize the input using the processor's image_mean and image_std
        norm_mean = torch.tensor(self.processor.image_mean, device=device).view(1, -1, 1, 1)
        norm_std = torch.tensor(self.processor.image_std, device=device).view(1, -1, 1, 1)
        inputs = (frames - norm_mean) / norm_std
        
        # Forward the preprocessed frames through the depth model
        model_output = self.depth_model(inputs)
        if hasattr(model_output, "predicted_depth"):
            depth_pred = model_output.predicted_depth  # Expected shape: (b*f, 1, H_out, W_out)
        else:
            depth_pred = model_output["depth"]
        
        # If the output is 3D, unsqueeze to add a channel dimension
        if depth_pred.dim() == 3:
            depth_pred = depth_pred.unsqueeze(1)
        
        # Interpolate depth predictions to match the original image size (h, w)
        depth_pred = F.interpolate(depth_pred, size=(h, w), mode="bilinear", align_corners=False)
        
        # Apply sigmoid activation to obtain an initial depth map in [0, 1]
        depth_initial = depth_pred.sigmoid()
        
        # Pass the initial depth map through the residual refinement block to learn a correction
        depth_refined = self.refinement_block(depth_initial)
        
        # Use the refined depth map as the final output
        depths = depth_refined.view(b, f, h, w)
        
        # Generate dummy backward correspondence weights (or use a learnable mechanism)
        backward_weights = torch.ones((b, f - 1, h, w), device=device)
        if self.cfg.weight_sensitivity is not None:
            backward_weights = (self.cfg.weight_sensitivity * self.weights).sigmoid()
            backward_weights = backward_weights[None]
        
        return BackboneOutput(depths, backward_weights)
