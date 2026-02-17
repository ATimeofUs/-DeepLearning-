from dataclasses import dataclass
from typing import Optional
from PIL import Image


@dataclass
class StageContext:
    """Shared data passed between pipeline stages.

    Fields are optional when not yet produced by a stage. Stages should read
    and update fields on the same shared context instance.
    """

    # Original inputs
    input_image: Image.Image
    input_path: str

    # Current processed image (stage may update this)
    current_image: Image.Image

    # Canvas / masks / pose artifacts
    canvas: Optional[Image.Image] = None
    
    # expand_mask 是扩展区域的蒙版，person_mask 是人物区域的蒙版
    expand_mask: Optional[Image.Image] = None
    pose_image: Optional[Image.Image] = None
    pose_canvas: Optional[Image.Image] = None

    # Debug / runtime
    debug_dir: Optional[str] = None

    # Prompts
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None

