from abc import ABC, abstractmethod
import numpy as np
import cv2
from dataclasses import replace
from PIL import Image, ImageFilter
from diffusers.utils import load_image
from typing import Protocol, runtime_checkable, Optional, Any

from .stage_context import StageContext


# =========================================================
# Protocol
# =========================================================


@runtime_checkable
class StageOwner(Protocol):
    pose_getter: Optional[Any]
    pipe_inpaint: Optional[Any]
    pipe_canny: Optional[Any]
    pipe_openpose: Optional[Any]

    def _lazy_load_models(self) -> None: ...


# =========================================================
# Base Stage
# =========================================================


class Stage(ABC):
    @abstractmethod
    def process(self, context: StageContext) -> StageContext:
        pass


# =========================================================
# Utils
# =========================================================


def _ceil_to(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def calculate_wh(h: int, w: int, ratio: float = 10 / 16, align: int = 64):
    if h / w < ratio:
        new_w = w
        new_h = int(round(w * ratio))
    else:
        new_h = h
        new_w = int(round(h / ratio))

    new_w = _ceil_to(new_w, align)
    new_h = _ceil_to(new_h, align)
    return new_h, new_w


# common keypoint connections used to draw skeleton lines
CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def expand_to_canvas(image: Image.Image, new_size, paste_pos):
    new_w, new_h = new_size
    if image.mode == "RGB":
        bg = (255, 255, 255)
    elif image.mode == "L":
        bg = 0
    else:
        bg = None

    canvas = Image.new(image.mode, (new_w, new_h), bg)
    canvas.paste(image, paste_pos)
    return canvas


def make_canny_control(image: Image.Image, low=120, high=220, blur_sigma=0.8):
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    edges = cv2.Canny(gray, low, high)
    edges_rgb = np.stack([edges] * 3, axis=2)
    return Image.fromarray(edges_rgb, mode="RGB")


# =========================================================
# Stage 0: Pose + Convex Hull Mask
# =========================================================


class Stage0Pose(Stage):
    def __init__(self, owner: StageOwner, padding: int = 30):
        self.owner = owner
        self.padding = padding

    def process(self, context: StageContext) -> StageContext:

        image = load_image(context.input_path).convert("RGB")
        img_np = np.array(image)

        keypoints, scores = self.owner.pose_getter.get_keypoints(img_np)

        w, h = image.size

        if keypoints is None or len(keypoints) == 0:
            pose_image = Image.new("RGB", (w, h), (0, 0, 0))
            return replace(context, pose_image=pose_image)

        pose_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Stage0 仅绘制关键点标记；骨架线绘制已移至 Stage3（更准确地在最终图像上重新检测并绘制）
        for person_idx, kp in enumerate(keypoints):
            for idx, point in enumerate(kp):
                if scores[person_idx][idx] > 0.3:
                    pt = tuple(point.astype(int))
                    cv2.circle(pose_canvas, pt, 4, (0, 255, 0), -1)

        pose_image = Image.fromarray(pose_canvas)
        return replace(context, pose_image=pose_image)


# =========================================================
# Stage 1: Base Inpaint
# =========================================================
class Stage1BaseInpaint(Stage):
    def __init__(
        self,
        owner: StageOwner,
        feather_radius=12,
        guidance_scale=7.5,
        num_inference_steps=35,
        strength=1.0,
    ):
        self.owner = owner
        self.feather_radius = feather_radius
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength

    def process(self, context: StageContext) -> StageContext:

        canvas, expand_mask, paste_pos = create_canvas_and_mask(
            context.input_path,
            feather_radius=self.feather_radius,
        )

        new_w, new_h = canvas.size

        pose_canvas = expand_to_canvas(context.pose_image, (new_w, new_h), paste_pos)

        # unify masks: use the expand_mask produced for the canvas as the single mask
        # expand_mask is already canvas-sized (new_w, new_h)

        if self.owner.pipe_inpaint is None:
            self.owner._lazy_load_models()

        result = self.owner.pipe_inpaint(
            prompt=context.prompt or "",
            negative_prompt=context.negative_prompt or "",
            image=canvas,
            mask_image=expand_mask,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            width=new_w,
            height=new_h,
        ).images[0]

        return replace(
            context,
            current_image=result,
            canvas=canvas,
            expand_mask=expand_mask,
            pose_canvas=pose_canvas,
        )


# =========================================================
# Stage 2: Canny Background Refine
# =========================================================


class Stage2CannyBackground(Stage):
    def __init__(
        self,
        owner: StageOwner,
        controlnet_conditioning_scale=0.75,
        guidance_scale=10.0,
        num_inference_steps=35,
        strength=0.80,
    ):
        self.owner = owner
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength

    def process(self, context: StageContext) -> StageContext:

        base_image = context.current_image
        w, h = base_image.size

        canny_control = make_canny_control(base_image)

        result = self.owner.pipe_canny(
            prompt=context.prompt or "",
            negative_prompt=context.negative_prompt or "",
            image=base_image,
            mask_image=context.expand_mask,
            control_image=canny_control,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            width=w,
            height=h,
        ).images[0]

        return replace(context, current_image=result)


# =========================================================
# Stage 3: OpenPose Refine (Re-detect Pose!)
# =========================================================


class Stage3OpenPoseRefine(Stage):
    def __init__(
        self,
        owner: StageOwner,
        controlnet_conditioning_scale=0.85,
        guidance_scale=6.5,
        num_inference_steps=40,
        strength=0.55,
    ):
        self.owner = owner
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength

    def process(self, context: StageContext) -> StageContext:

        if self.owner.pipe_openpose is None:
            self.owner._lazy_load_models()

        w, h = context.current_image.size

        # 重新检测 pose 并绘制标准 skeleton（线+点），作为 ControlNet 的 control_image
        img_np = np.array(context.current_image)
        keypoints, scores = self.owner.pose_getter.get_keypoints(img_np)

        pose_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if keypoints is not None:
            for person_idx, kp in enumerate(keypoints):
                # 先绘制骨架线（白色），符合 OpenPose ControlNet 常见输入
                for start_idx, end_idx in CONNECTIONS:
                    if start_idx < len(kp) and end_idx < len(kp):
                        if (
                            scores[person_idx][start_idx] > 0.3
                            and scores[person_idx][end_idx] > 0.3
                        ):
                            start = tuple(kp[start_idx].astype(int))
                            end = tuple(kp[end_idx].astype(int))
                            cv2.line(pose_canvas, start, end, (255, 255, 255), 3)

                # 再绘制关键点圆点以增强控制信号
                for idx, point in enumerate(kp):
                    if scores[person_idx][idx] > 0.3:
                        pt = tuple(point.astype(int))
                        cv2.circle(pose_canvas, pt, 4, (255, 255, 255), -1)

        pose_canvas_pil = Image.fromarray(pose_canvas)

        result = self.owner.pipe_openpose(
            prompt=context.prompt or "",
            negative_prompt=context.negative_prompt or "",
            image=context.current_image,
            mask_image=context.expand_mask,
            control_image=pose_canvas_pil,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            width=w,
            height=h,
        ).images[0]

        return replace(context, current_image=result)


def create_canvas_and_mask(img_path, feather_radius=4, smart_fill=True):
    image = load_image(img_path).convert("RGB")

    max_h = 1600
    max_w = 2560

    w, h = image.size

    if h > max_h or w > max_w:
        scale = min(max_w / w, max_h / h)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, resample=Image.LANCZOS)
        w, h = image.size

    new_h, new_w = calculate_wh(h, w, ratio=10 / 16, align=64)

    p_x = (new_w - w) // 2
    p_y = (new_h - h) // 2

    # 智能填充：用边缘平均色填充，而不是全黑
    if smart_fill:
        img_np = np.array(image)
        top_color = img_np[0, :, :].mean(axis=0).astype(int)
        bottom_color = img_np[-1, :, :].mean(axis=0).astype(int)
        left_color = img_np[:, 0, :].mean(axis=0).astype(int)
        right_color = img_np[:, -1, :].mean(axis=0).astype(int)

        avg_color = ((top_color + bottom_color + left_color + right_color) / 4).astype(
            int
        )
        canvas = Image.new("RGB", (new_w, new_h), tuple(avg_color))
    else:
        canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))

    canvas.paste(image, (p_x, p_y))

    # mask: 白色=重绘区域, 黑色=保留区域
    mask = Image.new("L", (new_w, new_h), 255)
    keep = Image.new("L", (w, h), 0)
    mask.paste(keep, (p_x, p_y))

    # 羽化边缘
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    return canvas, mask, (p_x, p_y)
