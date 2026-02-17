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
from typing import Dict, Optional


@dataclass
class DatasetConfig:
    main_model_file: str = "/run/media/ping/TRAIN_DATA1/model/"
    diffusers_stable_diffusion_xl_inpainting_model: str = "pic/diffusers-stable-diffusion-xl-inpainting-1.0"
    animagine_xl: str = "pic/animagine-xl-3.0"
    control_model_canny: str = "pic/xinsir-controlnet-canny-sdxl-1.0"
    control_model_openpose: str = "pic/xinsir-controlnet-openpose-sdxl-1.0"
    vae_model: str = "pic/madebyollin-sdxl-vae-fp16-fix"

    def __init__(self, base: Optional[str] = None):
        # Decide base path: prefer provided `base`, else the annotated default.
        base_candidate = base or self.main_model_file
        alt_candidate = "/run/media/ping/TRAIN_DATA/model/"

        # If the primary candidate doesn't exist but an alternate does, use the alternate.
        if not os.path.exists(base_candidate):
            if os.path.exists(alt_candidate):
                base_candidate = alt_candidate
            else:
                raise FileNotFoundError(
                    f"模型路径错误 ❌ Neither the provided base path '{base_candidate}' nor the alternate '{alt_candidate}' exists."
                )

        self.main_model_file = base_candidate

        # Build full model paths by joining with the base path.
        self.diffusers_stable_diffusion_xl_inpainting_model = os.path.join(
            self.main_model_file, "pic", "diffusers-stable-diffusion-xl-inpainting-1.0"
        )
        self.animagine_xl = os.path.join(self.main_model_file, "pic", "animagine-xl-3.0")
        self.control_model_canny = os.path.join(
            self.main_model_file, "pic", "xinsir-controlnet-canny-sdxl-1.0"
        )
        self.control_model_openpose = os.path.join(
            self.main_model_file, "pic", "xinsir-controlnet-openpose-sdxl-1.0"
        )
        self.vae_model = os.path.join(self.main_model_file, "pic", "madebyollin-sdxl-vae-fp16-fix")

    
    @classmethod
    def from_env(cls) -> "DatasetConfig":
        env_base = os.environ.get("MAIN_MODEL_FILE")
        if env_base:
            return cls(base=env_base)
        return cls()

    def as_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


def get_default_config() -> DatasetConfig:
    """Return a default DatasetConfig instance (from environment if set)."""
    return DatasetConfig.from_env()
