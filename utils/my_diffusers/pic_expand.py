import os
import inspect
import cv2
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Optional, Mapping, Any, Protocol, runtime_checkable, Literal
import numpy as np
from PIL import Image, ImageFilter
from diffusers.utils import load_image
from .control_image_get import (
    RtmlibPoseGet,
    draw_pose_skeleton_image,
    get_depth_zoe,
    get_canny_control,
    get_tile_control,
)
from utils.config import get_default_config
from .stage_context import StageContext

# pipeline loaders moved to separate module
from .pipeline_loaders import (
    load_inpaint_pipe,
    load_multi_controlnet_inpaint_pipe,
)

# =========================================================
# 模型选择枚举
# =========================================================
class ModelType:
    """模型类型常量"""
    BASE = 0      # 使用 base_model
    ANIME = 1     # 使用 animagine_model


# =========================================================
# Protocol
# =========================================================
@runtime_checkable
class StageDependencies(Protocol):
    """Dependencies required by pipeline stages.
    stage必须依赖于一个抽象的 StageDependencies 接口，而不是直接依赖于 ThreeStageOutpaint 类。
    这种设计允许我们在测试阶段时注入一个简单的 mock 对象，而不需要实例化整个 ThreeStageOutpaint 类，
    从而实现更轻量级和更快的单元测试。
    """
    cfg: Any

    def get_inpaint_pipe(self, model_type: int = ModelType.BASE) -> Any: ...
    def get_depth_canny_pipe(self, model_type: int = ModelType.BASE) -> Any: ...
    def get_tile_pose_pipe(self, model_type: int = ModelType.BASE) -> Any: ...
    def get_pose_getter(self) -> Any: ...


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


# =========================================================
# Stage 1: Base Inpaint
# =========================================================
class Stage1BaseInpaint(Stage):
    def __init__(
        self,
        deps: StageDependencies,
        feather_radius=12,
        guidance_scale=7.5,
        num_inference_steps=35,
        strength=1.0,
        model_type: int = ModelType.BASE,  # 新增：模型选择参数
    ):
        self.deps = deps
        self.feather_radius = feather_radius
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.model_type = model_type
        self._validate_model_type()

    def _validate_model_type(self):
        """验证模型类型参数"""
        if self.model_type not in (ModelType.BASE, ModelType.ANIME):
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Use ModelType.BASE(0) or ModelType.ANIME(1)"
            )

    def process(self, context: StageContext) -> StageContext:
        canvas, expand_mask, _paste_pos = create_canvas_and_mask(
            context.input_path, feather_radius=self.feather_radius, fill_way=0
        )
        new_w, new_h = canvas.size
        pipe_inpaint = self.deps.get_inpaint_pipe(model_type=self.model_type)
        
        print(f"📌 Stage1 使用模型: {'ANIME' if self.model_type == ModelType.ANIME else 'BASE'}")
        
        result = pipe_inpaint(
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
        )


# =========================================================
# Stage 2: Depth + Canny (refine background)
# =========================================================
class Stage2DepthCanny(Stage):
    """使用 Depth + Canny 对原图细化，通过 Stage1 生成的 canny depth 图片进行细化（仅在 mask 区域）。
    
    新增功能：
    - model_type 参数选择 base 或 anime 模型
    - image_source 参数选择输入到模型的图片来源（canvas 或 current_image）
    - mask_mode 参数选择使用的 mask（0=expand_mask, 1=全白 mask）
    """
    def __init__(
        self,
        deps: StageDependencies,
        controlnet_conditioning_scale=None,  # [depth_scale, canny_scale]
        guidance_scale=7.0,
        num_inference_steps=40,
        strength=0.75,
        depth_fn=get_depth_zoe,
        canny_fn=get_canny_control,
        model_type: int = ModelType.BASE,  # 新增：模型选择参数
        image_source: Literal["canvas", "current"] = "current",  # 新增：选择输入图片源
        mask_mode: int = 0,  # 新增：mask 模式，0=expand_mask, 1=全白 mask
    ):
        self.deps = deps
        if controlnet_conditioning_scale is None:
            controlnet_conditioning_scale = [0.85, 0.70]
        elif not isinstance(controlnet_conditioning_scale, list):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * 2
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.depth_fn = depth_fn
        self.canny_fn = canny_fn
        self.model_type = model_type
        self.image_source = image_source
        self.mask_mode = mask_mode
        self._validate_params()

    def _validate_params(self):
        """验证参数"""
        if self.model_type not in (ModelType.BASE, ModelType.ANIME):
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Use ModelType.BASE(0) or ModelType.ANIME(1)"
            )
        if self.image_source not in ("canvas", "current"):
            raise ValueError(
                f"Invalid image_source: {self.image_source}. "
                f"Use 'canvas' or 'current'"
            )
        if self.mask_mode not in (0, 1):
            raise ValueError(
                f"Invalid mask_mode: {self.mask_mode}. "
                f"Use 0(expand_mask) or 1(全白 mask)"
            )

    def process(self, context: StageContext) -> StageContext:
        pipe_cn_depth_canny = self.deps.get_depth_canny_pipe(
            model_type=self.model_type
        )
        
        # 根据 image_source 选择输入到模型的图片
        if self.image_source == "canvas":
            input_image = context.canvas if context.canvas is not None else context.current_image
            print(f"📌 Stage2 输入图片来源: canvas")
        else:  # "current"
            input_image = context.current_image
            print(f"📌 Stage2 输入图片来源: current_image")
        
        w, h = input_image.size

        depth_control = self.depth_fn(
            input_image, model_dir=self.deps.cfg.lllyasviel_model
        )
        depth_control = depth_control.resize((w, h), resample=Image.BILINEAR)
        canny_control = self.canny_fn(input_image)

        # 根据 mask_mode 选择 mask
        if self.mask_mode == 1:
            # 全白 mask - 模型可以重绘整个图像
            mask_image = Image.new("L", (w, h), 255)
            print(f"🎨 Stage2 Mask 模式: 全白 mask（重绘模式）")
        else:
            # expand_mask - 只在扩展区域生成
            mask_image = context.expand_mask
            print(f"🎨 Stage2 Mask 模式: expand_mask（扩展区域）")

        print(f"📌 Stage2 使用模型: {'ANIME' if self.model_type == ModelType.ANIME else 'BASE'}")
        print(f"📊 Depth 权重: {self.controlnet_conditioning_scale[0]}")
        print(f"📊 Canny 权重: {self.controlnet_conditioning_scale[1]}")

        if context.debug_dir:
            depth_path = os.path.join(
                context.debug_dir, "debug_stage2_depth_control.png"
            )
            canny_path = os.path.join(
                context.debug_dir, "debug_stage2_canny_control.png"
            )
            depth_control.save(depth_path)
            canny_control.save(canny_path)
            if self.mask_mode == 1:
                mask_path = os.path.join(
                    context.debug_dir, "debug_stage2_mask_white.png"
                )
            else:
                mask_path = os.path.join(
                    context.debug_dir, "debug_stage2_mask_expand.png"
                )
            mask_image.save(mask_path)
            print(f"🛠️ Stage2 ControlNet 输入已保存到 debug 目录: {context.debug_dir}")

        result = pipe_cn_depth_canny(
            prompt=context.prompt or "",
            negative_prompt=context.negative_prompt or "",
            image=input_image,
            mask_image=mask_image,
            control_image=[depth_control, canny_control],
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            width=w,
            height=h,
        ).images[0]

        return replace(context, current_image=result)


# =========================================================
# Stage 3: Tile + Pose (refine background & people)
# =========================================================
class Stage3TilePose(Stage):
    """使用 Tile + Pose 对背景和背景人物进行进一步细化（全图区域）。
    
    新增功能：
    - model_type 参数选择 base 或 anime 模型
    - image_source 参数选择输入到模型的图片来源（canvas 或 current_image）
    - mask_mode 参数选择使用的 mask（0=expand_mask, 1=全白 mask）
    """
    def __init__(
        self,
        deps: StageDependencies,
        controlnet_conditioning_scale=None,  # [tile_scale, pose_scale]
        guidance_scale=7.0,
        num_inference_steps=45,
        strength=0.65,
        pose_score_threshold: float = 0.3,
        pose_from: str = "canny",  # canny|image
        tile_fn=get_tile_control,
        canny_fn=get_canny_control,
        pose_draw_fn=draw_pose_skeleton_image,
        model_type: int = ModelType.BASE,  # 新增：模型选择参数
        image_source: Literal["canvas", "current"] = "current",  # 新增：选择输入图片源
        mask_mode: int = 0,  # 新增：mask 模式，0=expand_mask, 1=全白 mask
    ):
        self.deps = deps
        if controlnet_conditioning_scale is None or not isinstance(
            controlnet_conditioning_scale, list
        ):
            raise ValueError("controlnet_conditioning_scale must be a list of two floats")
        if len(controlnet_conditioning_scale) != 2:
            raise ValueError("controlnet_conditioning_scale must be a list of two floats")
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.pose_score_threshold = pose_score_threshold
        self.pose_from = pose_from
        self.tile_fn = tile_fn
        self.canny_fn = canny_fn
        self.pose_draw_fn = pose_draw_fn
        self.model_type = model_type
        self.image_source = image_source
        self.mask_mode = mask_mode
        self._validate_params()

    def _validate_params(self):
        """验证参数"""
        if self.model_type not in (ModelType.BASE, ModelType.ANIME):
            raise ValueError(
                f"Invalid model_type: {self.model_type}. "
                f"Use ModelType.BASE(0) or ModelType.ANIME(1)"
            )
        if self.image_source not in ("canvas", "current"):
            raise ValueError(
                f"Invalid image_source: {self.image_source}. "
                f"Use 'canvas' or 'current'"
            )
        if self.mask_mode not in (0, 1):
            raise ValueError(
                f"Invalid mask_mode: {self.mask_mode}. "
                f"Use 0(expand_mask) or 1(全白 mask)"
            )

    def process(self, context: StageContext) -> StageContext:
        pipe_cn_tile_pose = self.deps.get_tile_pose_pipe(model_type=self.model_type)
        pose_getter = self.deps.get_pose_getter()
        
        # 根据 image_source 选择输入到模型的图片
        if self.image_source == "canvas":
            input_image = context.canvas if context.canvas is not None else context.current_image
            print(f"📌 Stage3 输入图片来源: canvas")
        else:  # "current"
            input_image = context.current_image
            print(f"📌 Stage3 输入图片来源: current_image")
        
        w, h = input_image.size

        tile_control = self.tile_fn(input_image)
        canny_for_pose = self.canny_fn(input_image)
        pose_src = (
            canny_for_pose if self.pose_from == "canny" else input_image
        )
        keypoints, scores = pose_getter.get_keypoints(
            np.array(pose_src.convert("RGB"))
        )
        pose_control = self.pose_draw_fn(
            keypoints,
            scores,
            size=(w, h),
            score_threshold=self.pose_score_threshold,
            line_color_bgr=(255, 255, 255),
            line_thickness=3,
            point_color_bgr=(255, 255, 255),
            point_radius=4,
            background_bgr=(0, 0, 0),
        )

        # 根据 mask_mode 选择 mask
        if self.mask_mode == 1:
            # 全白 mask - 模型可以重绘整个图像
            mask_image = Image.new("L", (w, h), 255)
            print(f"🎨 Stage3 Mask 模式: 全白 mask（重绘模式）")
        else:
            # expand_mask - 只在扩展区域生成
            mask_image = context.expand_mask
            print(f"🎨 Stage3 Mask 模式: expand_mask（扩展区域）")

        print(f"📌 Stage3 使用模型: {'ANIME' if self.model_type == ModelType.ANIME else 'BASE'}")
        print(f"📊 Tile 权重: {self.controlnet_conditioning_scale[0]}")
        print(f"📊 Pose 权重: {self.controlnet_conditioning_scale[1]}")

        if context.debug_dir:
            tile_path = os.path.join(context.debug_dir, "debug_stage3_tile_control.png")
            pose_path = os.path.join(context.debug_dir, "debug_stage3_pose_control.png")
            tile_control.save(tile_path)
            pose_control.save(pose_path)
            if self.mask_mode == 1:
                mask_path = os.path.join(
                    context.debug_dir, "debug_stage3_mask_white.png"
                )
            else:
                mask_path = os.path.join(
                    context.debug_dir, "debug_stage3_mask_expand.png"
                )
            mask_image.save(mask_path)
            print(f"🛠️ Stage3 ControlNet 输入已保存到 debug 目录: {context.debug_dir}")

        result = pipe_cn_tile_pose(
            prompt=context.prompt or "",
            negative_prompt=context.negative_prompt or "",
            image=input_image,
            mask_image=mask_image,
            control_image=[tile_control, pose_control],
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            width=w,
            height=h,
        ).images[0]

        return replace(context, current_image=result)


# =========================================================
# Canvas and Mask Creation
# =========================================================
def create_canvas_and_mask(
    img_path, feather_radius=0, fill_way: int = 0, depth_img: Image.Image = None
):
    """
    创建扩展画布和掩码

    :return:
        - canvas: 扩展后的画布图，大小为 (new_w, new_h)，原图位于中心，周围填充
        - expand_mask_soft: 羽化后的扩展掩码，白色区域为可生成区域，黑色区域为保留区域
        - (p_x, p_y): 原图在画布上的粘贴位置坐标

    :param img_path: 输入图像路径
    :param feather_radius: 羽化半径
    :param fill_way: 填充方式（0=平均颜色, 1=随机噪声, 2=全黑）
    :param depth_img: 深度图，用于调整背景亮度
    """
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
    if fill_way == 0:
        img_np = np.array(image)
        top_color = img_np[0, :, :].mean(axis=0).astype(int)
        bottom_color = img_np[-1, :, :].mean(axis=0).astype(int)
        left_color = img_np[:, 0, :].mean(axis=0).astype(int)
        right_color = img_np[:, -1, :].mean(axis=0).astype(int)
        avg_color = ((top_color + bottom_color + left_color + right_color) / 4).astype(
            int
        )
        canvas = Image.new("RGB", (new_w, new_h), tuple(avg_color))
    elif fill_way == 1:
        canvas = Image.fromarray(
            np.random.randint(0, 256, (new_h, new_w, 3), dtype=np.uint8)
        )
    else:
        canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))

    if depth_img is not None:
        fac = np.array(depth_img) / 255.0  # (new_h, new_w)
        fac = np.expand_dims(fac, axis=2)  # (new_h, new_w, 1)
        canvas_np = np.array(canvas).astype(np.float32)
        canvas_np = canvas_np * fac
        canvas = Image.fromarray(canvas_np.astype(np.uint8))

    canvas.paste(image, (p_x, p_y))

    # hard masks
    # expand_mask_hard: 白色=可生成(扩展)区域, 黑色=保留(原图)区域
    expand_mask_hard = Image.new("L", (new_w, new_h), 255)
    keep_area = Image.new("L", (w, h), 0)
    expand_mask_hard.paste(keep_area, (p_x, p_y))

    # feathered mask for inpaint, but must never leak into keep region
    expand_mask_soft = expand_mask_hard
    if feather_radius and feather_radius > 0:
        blurred = expand_mask_hard.filter(
            ImageFilter.GaussianBlur(radius=feather_radius)
        )
        # clamp: keep-region stays exactly 0
        soft_np = np.array(blurred, dtype=np.uint8)
        hard_np = np.array(expand_mask_hard, dtype=np.uint8)
        soft_np = np.minimum(soft_np, hard_np)
        expand_mask_soft = Image.fromarray(soft_np, mode="L")

    return canvas, expand_mask_soft, (p_x, p_y)


# =========================================================
# ThreeStageOutpaint Main Class
# =========================================================
class ThreeStageOutpaint:
    """
    三阶段 Outpaint 管道封装

    新增功能：
    - 各个 Stage 支持 model_type 参数，可以数字方式指定 (0=base, 1=anime)
    - Stage2 集成了 Stage3 的 rebuild 功能
    - 支持灵活的 stage_params 覆盖
    """

    def __init__(
        self,
        cfg=None,
        device: str = "cuda",
        pose_getter: Optional[object] = None,
        pipe_inpaint: Optional[object] = None,
        lazy_load: bool = True,
        stage_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        self.cfg = cfg or get_default_config()
        self.device = device

        # model ids/paths from config
        self.animagine_model = self.cfg.animagine_xl
        self.animagine_model_4 = self.cfg.animagine_xl_4
        self.base_model = self.cfg.diffusers_stable_diffusion_xl_inpainting_model
        self.ctl_model_canny = self.cfg.control_model_canny
        self.ctl_model_depth = self.cfg.control_model_depth
        self.ctl_model_openpose = self.cfg.control_model_openpose
        self.ctl_model_tile = self.cfg.control_model_tile
        self.vae_model = self.cfg.vae_model
        self.lora_model = self.cfg.lora_model

        # optionally injected instances
        self.pose_getter = pose_getter
        self.pipe_inpaint = pipe_inpaint
        self.pipe_inpaint_anime = None  # 新增：anime inpaint pipeline
        self.pipe_cn_depth_canny = None
        self.pipe_cn_depth_canny_anime = None  # 新增
        self.pipe_cn_tile_pose = None
        self.pipe_cn_tile_pose_anime = None  # 新增

        # lazy loading control
        self._models_loaded = False
        self._lazy = lazy_load
        if not self._lazy:
            self._lazy_load_models()

        # 字典化 stage 参数
        stage_params = dict(stage_params or {})
        deps = _ThreeStageOutpaintDeps(self)

        # 三阶段流程
        self.stages = [
            Stage1BaseInpaint(
                deps,
                **self._stage_kwargs(
                    "stage1",
                    Stage1BaseInpaint,
                    feather_radius=0,
                    guidance_scale=8.5,
                    num_inference_steps=30,
                    strength=1.0,
                    model_type=ModelType.BASE,
                    stage_params=stage_params,
                ),
            ),
            Stage2DepthCanny(
                deps,
                **self._stage_kwargs(
                    "stage2",
                    Stage2DepthCanny,
                    controlnet_conditioning_scale=[0.85, 0.70],
                    guidance_scale=7.0,
                    num_inference_steps=40,
                    strength=0.75,
                    model_type=ModelType.BASE,
                    image_source="current",
                    mask_mode=0,
                    stage_params=stage_params,
                ),
            ),
            Stage3TilePose(
                deps,
                **self._stage_kwargs(
                    "stage3",
                    Stage3TilePose,
                    controlnet_conditioning_scale=[0.60, 0.85],
                    guidance_scale=7.0,
                    num_inference_steps=45,
                    strength=0.65,
                    model_type=ModelType.BASE,
                    image_source="current",
                    mask_mode=0,
                    stage_params=stage_params,
                ),
            ),
        ]

        unknown_stage_keys = set(stage_params.keys()) - {
            "stage1",
            "stage2",
            "stage3",
        }
        if unknown_stage_keys:
            raise ValueError(
                f"Unknown stage keys in stage_params: {sorted(unknown_stage_keys)}"
            )

    @staticmethod
    def _stage_kwargs(
        stage_name: str,
        stage_cls: type,
        /,
        stage_params: Mapping[str, Mapping[str, Any]],
        **defaults: Any,
    ) -> dict[str, Any]:
        """生成 stage 参数，支持覆盖默认值"""
        overrides = dict(stage_params.get(stage_name, {}) or {})

        # 验证覆盖参数的有效性
        sig = inspect.signature(stage_cls.__init__)
        valid = {p.name for p in sig.parameters.values()} - {"self", "owner", "deps"}
        unknown = set(overrides.keys()) - valid
        if unknown:
            raise ValueError(
                f"Unknown params for {stage_name} ({stage_cls.__name__}): {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}"
            )

        merged = dict(defaults)
        merged.update(overrides)
        return merged

    def _lazy_load_models(self):
        """懒加载所有模型"""
        if self._models_loaded:
            return

        print("🚀 正在准备模型路径和资源...")
        print("animagine_model:", self.animagine_model)
        print("base_model:", self.base_model)
        print("ctl_model_canny:", self.ctl_model_canny)
        print("ctl_model_depth:", self.ctl_model_depth)
        print("ctl_model_openpose:", self.ctl_model_openpose)
        print("vae_model:", self.vae_model)

        print("\n🚀 正在加载模型...")

        if self.pose_getter is None:
            print("📌 加载 Pose 检测器...")
            self.pose_getter = RtmlibPoseGet()

        if self.pipe_inpaint is None:
            print("📌 加载 Base Inpaint Pipeline...")
            self.pipe_inpaint = load_inpaint_pipe(self.base_model, self.vae_model)

        if self.pipe_inpaint_anime is None:
            print("📌 加载 Anime Inpaint Pipeline...")
            self.pipe_inpaint_anime = load_inpaint_pipe(
                self.animagine_model, self.vae_model
            )

        if self.pipe_cn_depth_canny is None:
            print("📌 加载 Stage2 Depth+Canny ControlNet Pipeline (Base)...")
            self.pipe_cn_depth_canny = load_multi_controlnet_inpaint_pipe(
                base_model=self.base_model,
                control_models=[
                    self.ctl_model_depth,
                    self.ctl_model_canny,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        if self.pipe_cn_depth_canny_anime is None:
            print("📌 加载 Stage2 Depth+Canny ControlNet Pipeline (Anime)...")
            self.pipe_cn_depth_canny_anime = load_multi_controlnet_inpaint_pipe(
                base_model=self.animagine_model,
                control_models=[
                    self.ctl_model_depth,
                    self.ctl_model_canny,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        if self.pipe_cn_tile_pose is None:
            print("📌 加载 Stage3 Tile+Pose ControlNet Pipeline (Base)...")
            self.pipe_cn_tile_pose = load_multi_controlnet_inpaint_pipe(
                base_model=self.base_model,
                control_models=[
                    self.ctl_model_tile,
                    self.ctl_model_openpose,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        if self.pipe_cn_tile_pose_anime is None:
            print("📌 加载 Stage3 Tile+Pose ControlNet Pipeline (Anime)...")
            self.pipe_cn_tile_pose_anime = load_multi_controlnet_inpaint_pipe(
                base_model=self.animagine_model,
                control_models=[
                    self.ctl_model_tile,
                    self.ctl_model_openpose,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        self._models_loaded = True
        print("✅ 所有模型加载完成！\n")

    def execute_once(
        self,
        input_path: str = None,
        tmp_dir: str = None,
        debug: bool = False,
        prompt: str = None,
        negative_prompt: str = None,
    ):
        """
        执行三阶段 outpaint

        :param input_path: 输入图像路径
        :param tmp_dir: 临时目录（保存中间结果）
        :param debug: 是否保存调试信息
        :param prompt: 正向提示词
        :param negative_prompt: 负向提示词
        """
        self._lazy_load_models()
        os.makedirs(tmp_dir, exist_ok=True)
        input_image = load_image(input_path).convert("RGB")

        context = StageContext(
            input_image=input_image,
            input_path=input_path,
            current_image=input_image,
            debug_dir=tmp_dir if debug else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        # Stage 1
        print("\n=== Stage 1: Base Inpaint 扩展背景 ===")
        context = self.stages[0].process(context)
        if debug:
            stage1_path = os.path.join(tmp_dir, "stage1_inpaint.png")
            context.current_image.save(stage1_path)
            print(f"Stage1 保存: {stage1_path}")
            if context.canvas is not None:
                stage1_canvas_path = os.path.join(tmp_dir, "stage1_canvas.png")
                context.canvas.save(stage1_canvas_path)
                print(f"Stage1 Canvas 保存: {stage1_canvas_path}")

        # Stage 2
        print("\n=== Stage 2: Depth + Canny 细化背景 ===")
        context = self.stages[1].process(context)
        if debug:
            stage2_path = os.path.join(tmp_dir, "stage2_depth_canny.png")
            context.current_image.save(stage2_path)
            print(f"Stage2 保存: {stage2_path}")

        # Stage 3
        print("\n=== Stage 3: Tile + Pose 深度细化 ===")
        context = self.stages[2].process(context)
        if debug:
            stage3_path = os.path.join(tmp_dir, "stage3_tile_pose.png")
            context.current_image.save(stage3_path)
            print(f"Stage3 保存: {stage3_path}")

        return context


class _ThreeStageOutpaintDeps:
    """适配器：将 ThreeStageOutpaint 暴露为 StageDependencies，不泄露内部细节"""

    def __init__(self, outpainter: ThreeStageOutpaint):
        self._outpainter = outpainter

    @property
    def cfg(self) -> Any:
        return self._outpainter.cfg

    def get_inpaint_pipe(self, model_type: int = ModelType.BASE) -> Any:
        self._outpainter._lazy_load_models()
        if model_type == ModelType.ANIME:
            return self._outpainter.pipe_inpaint_anime
        else:
            return self._outpainter.pipe_inpaint

    def get_depth_canny_pipe(self, model_type: int = ModelType.BASE) -> Any:
        self._outpainter._lazy_load_models()
        if model_type == ModelType.ANIME:
            return self._outpainter.pipe_cn_depth_canny_anime
        else:
            return self._outpainter.pipe_cn_depth_canny

    def get_tile_pose_pipe(self, model_type: int = ModelType.BASE) -> Any:
        self._outpainter._lazy_load_models()
        if model_type == ModelType.ANIME:
            return self._outpainter.pipe_cn_tile_pose_anime
        else:
            return self._outpainter.pipe_cn_tile_pose

    def get_pose_getter(self) -> Any:
        self._outpainter._lazy_load_models()
        return self._outpainter.pose_getter


# =========================================================
# TwoStageOutpaint
# =========================================================
class TwoStageOutpaint:
    """
    两阶段 Outpaint（移除 Stage1）

    新增功能：
    - 各个 Stage 支持 model_type 参数
    - Stage2 集成了 rebuild 功能
    """

    def __init__(
        self,
        cfg=None,
        device: str = "cuda",
        pose_getter: Optional[object] = None,
        lazy_load: bool = True,
        stage_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        self.cfg = cfg or get_default_config()
        self.device = device

        # model ids/paths from config
        self.animagine_model = self.cfg.animagine_xl
        self.base_model = self.cfg.diffusers_stable_diffusion_xl_inpainting_model
        self.ctl_model_canny = self.cfg.control_model_canny
        self.ctl_model_depth = self.cfg.control_model_depth
        self.ctl_model_openpose = self.cfg.control_model_openpose
        self.ctl_model_tile = self.cfg.control_model_tile
        self.vae_model = self.cfg.vae_model
        self.lora_model = self.cfg.lora_model

        # optionally injected instances
        self.pose_getter = pose_getter
        self.pipe_cn_depth_canny = None
        self.pipe_cn_depth_canny_anime = None
        self.pipe_cn_tile_pose = None
        self.pipe_cn_tile_pose_anime = None

        # lazy loading control
        self._models_loaded = False
        self._lazy = lazy_load
        if not self._lazy:
            self._lazy_load_models()

        # 字典化 stage 参数
        stage_params = dict(stage_params or {})
        deps = _TwoStageOutpaintDeps(self)

        self.stages = [
            Stage2DepthCanny(
                deps,
                **ThreeStageOutpaint._stage_kwargs(
                    "stage2",
                    Stage2DepthCanny,
                    controlnet_conditioning_scale=[0.85, 0.70],
                    guidance_scale=7.0,
                    num_inference_steps=40,
                    strength=0.75,
                    model_type=ModelType.BASE,
                    image_source="current",
                    mask_mode=0,
                    stage_params=stage_params,
                ),
            ),
            Stage3TilePose(
                deps,
                **ThreeStageOutpaint._stage_kwargs(
                    "stage3",
                    Stage3TilePose,
                    controlnet_conditioning_scale=[0.60, 0.85],
                    guidance_scale=7.0,
                    num_inference_steps=45,
                    strength=0.65,
                    model_type=ModelType.BASE,
                    image_source="current",
                    mask_mode=0,
                    stage_params=stage_params,
                ),
            ),
        ]

    def _lazy_load_models(self):
        """懒加载模型"""
        if self._models_loaded:
            return

        print("🚀 [TwoStage] 正在准备模型路径和资源...")

        if self.pose_getter is None:
            print("📌 加载 Pose 检测器...")
            self.pose_getter = RtmlibPoseGet()

        if self.pipe_cn_depth_canny is None:
            print("📌 加载 Stage2 Depth+Canny ControlNet Pipeline (Base)...")
            self.pipe_cn_depth_canny = load_multi_controlnet_inpaint_pipe(
                base_model=self.base_model,
                control_models=[
                    self.ctl_model_depth,
                    self.ctl_model_canny,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        if self.pipe_cn_depth_canny_anime is None:
            print("📌 加载 Stage2 Depth+Canny ControlNet Pipeline (Anime)...")
            self.pipe_cn_depth_canny_anime = load_multi_controlnet_inpaint_pipe(
                base_model=self.animagine_model,
                control_models=[
                    self.ctl_model_depth,
                    self.ctl_model_canny,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        if self.pipe_cn_tile_pose is None:
            print("📌 加载 Stage3 Tile+Pose ControlNet Pipeline (Base)...")
            self.pipe_cn_tile_pose = load_multi_controlnet_inpaint_pipe(
                base_model=self.base_model,
                control_models=[
                    self.ctl_model_tile,
                    self.ctl_model_openpose,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        if self.pipe_cn_tile_pose_anime is None:
            print("📌 加载 Stage3 Tile+Pose ControlNet Pipeline (Anime)...")
            self.pipe_cn_tile_pose_anime = load_multi_controlnet_inpaint_pipe(
                base_model=self.animagine_model,
                control_models=[
                    self.ctl_model_tile,
                    self.ctl_model_openpose,
                ],
                vae_model=self.vae_model,
                lora_model=self.lora_model,
            )

        self._models_loaded = True
        print("✅ 所有模型加载完成！\n")

    def execute_once(
        self,
        input_path: str,
        prepared_image: Image.Image,
        tmp_dir: str = None,
        debug: bool = False,
        prompt: str = None,
        negative_prompt: str = None,
    ):
        """
        执行两阶段 outpaint

        :param input_path: 原始图像路径
        :param prepared_image: 预处理后的图像（Stage 0 产物）
        :param tmp_dir: 临时目录
        :param debug: 是否保存调试信息
        :param prompt: 正向提示词
        :param negative_prompt: 负向提示词
        """
        self._lazy_load_models()
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)

        # 计算目标尺寸
        orig_img = load_image(input_path)
        w, h = orig_img.size
        max_h = 1600
        max_w = 2560
        if h > max_h or w > max_w:
            scale = min(max_w / w, max_h / h)
            w = int(w * scale)
            h = int(h * scale)
        new_h, new_w = calculate_wh(h, w, ratio=10 / 16, align=64)

        # 调整预处理图像大小
        if prepared_image.size != (new_w, new_h):
            print(
                f"⚠️ Resizing prepared image from {prepared_image.size} to {(new_w, new_h)}"
            )
            prepared_image = prepared_image.resize(
                (new_w, new_h), resample=Image.LANCZOS
            )
        prepared_image = prepared_image.convert("RGB")

        # 创建 context
        input_image_full = load_image(input_path).convert("RGB")
        canvas, expand_mask, _ = create_canvas_and_mask(input_path, fill_way=0)

        context = StageContext(
            input_image=input_image_full,
            input_path=input_path,
            canvas=canvas,
            expand_mask=expand_mask,
            current_image=prepared_image,
            debug_dir=tmp_dir if debug else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        # Stage 2
        print("\n=== Stage 2: Depth + Canny 细化背景 (TwoStage) ===")
        context = self.stages[0].process(context)
        if debug and tmp_dir:
            stage2_path = os.path.join(tmp_dir, "two_stage_stage2.png")
            context.current_image.save(stage2_path)
            print(f"Stage2 保存: {stage2_path}")

        # Stage 3
        print("\n=== Stage 3: Tile + Pose 深度细化 (TwoStage) ===")
        context = self.stages[1].process(context)
        if debug and tmp_dir:
            stage3_path = os.path.join(tmp_dir, "two_stage_stage3.png")
            context.current_image.save(stage3_path)
            print(f"Stage3 保存: {stage3_path}")

        return context


class _TwoStageOutpaintDeps:
    """适配器：将 TwoStageOutpaint 暴露为 StageDependencies"""

    def __init__(self, outpainter: TwoStageOutpaint):
        self._outpainter = outpainter

    @property
    def cfg(self) -> Any:
        return self._outpainter.cfg

    def get_inpaint_pipe(self, model_type: int = ModelType.BASE) -> Any:
        raise NotImplementedError(
            "TwoStageOutpaint does not support get_inpaint_pipe (Stage 1 is removed)."
        )

    def get_depth_canny_pipe(self, model_type: int = ModelType.BASE) -> Any:
        self._outpainter._lazy_load_models()
        if model_type == ModelType.ANIME:
            return self._outpainter.pipe_cn_depth_canny_anime
        else:
            return self._outpainter.pipe_cn_depth_canny

    def get_tile_pose_pipe(self, model_type: int = ModelType.BASE) -> Any:
        self._outpainter._lazy_load_models()
        if model_type == ModelType.ANIME:
            return self._outpainter.pipe_cn_tile_pose_anime
        else:
            return self._outpainter.pipe_cn_tile_pose

    def get_pose_getter(self) -> Any:
        self._outpainter._lazy_load_models()
        return self._outpainter.pose_getter