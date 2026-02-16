"""Configuration helpers for dataset and model paths.

Provide a simple `DatasetConfig` class holding common file paths used
across the project. Consumers can import and instantiate the class,
override values, or load values from environment variables.

Example:
    from utils.config.config import DatasetConfig
        cfg = DatasetConfig()
        print(cfg.base_model)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class DatasetConfig:
    diffusers_stable_diffusion_xl_inpainting_model: str = "/run/media/ping/TRAIN_DATA/model/pic/diffusers-stable-diffusion-xl-inpainting-1.0"
    animagine_xl: str = "/run/media/ping/TRAIN_DATA/model/pic/animagine-xl-3.0"
    control_model_canny: str = (
        "/run/media/ping/TRAIN_DATA/model/pic/xinsir-controlnet-canny-sdxl-1.0"
    )
    control_model_openpose: str = (
        "/run/media/ping/TRAIN_DATA/model/pic/xinsir-controlnet-openpose-sdxl-1.0"
    )
    vae_model: str = (
        "/run/media/ping/TRAIN_DATA/model/pic/madebyollin-sdxl-vae-fp16-fix"
    )
    input_path: str = "/home/ping/Downloads/100289305_p0_master1200.jpg"

    @classmethod
    def from_env(cls) -> "DatasetConfig":
        return cls()

    def as_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


def get_default_config() -> DatasetConfig:
    """Return a default DatasetConfig instance (from environment if set)."""
    return DatasetConfig.from_env()
