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

import os
from dataclasses import dataclass, asdict
from typing import List, ClassVar, Dict, Optional
import warnings


@dataclass
class DatasetConfig:
    diffusers_stable_diffusion_xl_inpainting_model: str = (
        "pic/diffusers-stable-diffusion-xl-inpainting-1.0"
    )
    animagine_xl: str = "pic/animagine-xl-3.0"
    animagine_xl_4: str = "pic/animagine-xl-4.0"

    control_model_canny: str = "pic/xinsir-controlnet-canny-sdxl-1.0"
    control_model_depth: str = "pic/xinsir-controlnet-depth-sdxl-1.0"
    control_model_openpose: str = "pic/xinsir-controlnet-openpose-sdxl-1.0"
    control_model_tile: str = "pic/xinsir-controlnet-tile-sdxl-1.0"

    lllyasviel_model: str = "pic/lllyasviel_Annotators"

    vae_model: str = "pic/madebyollin-sdxl-vae-fp16-fix"
    lora_model: str = (
        "pic/Linaqruf-style-enhancer-xl-lora/style-enhancer-xl.safetensors"
    )

    def __init__(self):
        # Decide base path: prefer explicit `base`, then env var, then known defaults.
        DEFAULT_BASE_CANDIDATES: List[str] = [
            "/run/media/ping/TRAIN_DATA1/model/",
            "/run/media/ping/TRAIN_DATA/model/",
        ]

        self.main_model_file = ""
        for t in DEFAULT_BASE_CANDIDATES:
            if os.path.exists(t):
                self.main_model_file = t
                break

        assert self.main_model_file != ""

        # Build full model paths by joining with the base path.
        self.diffusers_stable_diffusion_xl_inpainting_model = os.path.join(self.main_model_file, self.diffusers_stable_diffusion_xl_inpainting_model)
        self.animagine_xl = os.path.join(self.main_model_file, self.animagine_xl)
        self.animagine_xl_4 = os.path.join(self.main_model_file, self.animagine_xl_4)
        
        # ctl
        self.control_model_canny = os.path.join(self.main_model_file, self.control_model_canny)
        self.control_model_depth = os.path.join(self.main_model_file, self.control_model_depth)
        self.control_model_openpose = os.path.join(self.main_model_file, self.control_model_openpose)
        self.control_model_tile = os.path.join(self.main_model_file, self.control_model_tile)
        
        # vae 和 lora 模型路径
        self.vae_model = os.path.join(self.main_model_file, self.vae_model)
        self.lora_model = os.path.join(self.main_model_file, self.lora_model)
        self.lllyasviel_model = os.path.join(self.main_model_file, self.lllyasviel_model)

        # Warn about missing individual model files but do not block here.
        for k, v in self.as_dict().items():
            if v is None:
                warnings.warn(f"Configuration key {k} is None")
                continue
            if not os.path.exists(v):
                warnings.warn(f"Path for {k} does not exist: {v}")

    def as_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


def get_default_config() -> DatasetConfig:
    """Return a default DatasetConfig instance (from environment if set)."""
    return DatasetConfig()
