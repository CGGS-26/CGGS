from .backbone import Backbone
from .backbone_depth_anything import BackboneDepthAnything, BackboneDepthAnythingCfg
from .backbone_explicit_depth import BackboneExplicitDepth, BackboneExplicitDepthCfg
from .backbone_midas import BackboneMidas, BackboneMidasCfg

BACKBONES = {
    "depth_anything": BackboneDepthAnything,
    "explicit_depth": BackboneExplicitDepth,
    "midas": BackboneMidas,
}

BackboneCfg = BackboneDepthAnythingCfg | BackboneExplicitDepthCfg | BackboneMidasCfg


def get_backbone(
    cfg: BackboneCfg,
    num_frames: int | None,
    image_shape: tuple[int, int] | None,
) -> Backbone:
    return BACKBONES[cfg.name](cfg, num_frames, image_shape)
