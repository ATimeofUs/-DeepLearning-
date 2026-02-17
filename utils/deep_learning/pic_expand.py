import os
import numpy as np
import cv2
import torch
import shutil
from PIL import Image, ImageFilter
from .pose_get import RtmlibPoseGet
from typing import Optional

from diffusers.utils import load_image
from utils.config import get_default_config

# pipeline loaders moved to separate module
from .pipeline_loaders import load_inpaint_pipe, load_controlnet_inpaint_pipe


# =========================
# 1. 计算 16:10 画布
# =========================
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


# =========================
# 2. 创建画布 + 扩展区域 Mask
# =========================
def create_canvas_and_mask(img_path, feather_radius=12, smart_fill=True):
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
        # 获取原图四边的平均颜色
        img_np = np.array(image)
        top_color = img_np[0, :, :].mean(axis=0).astype(int)
        bottom_color = img_np[-1, :, :].mean(axis=0).astype(int)
        left_color = img_np[:, 0, :].mean(axis=0).astype(int)
        right_color = img_np[:, -1, :].mean(axis=0).astype(int)

        # 混合边缘颜色作为背景
        avg_color = ((top_color + bottom_color + left_color + right_color) / 4).astype(int)
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


# =========================
# 4. 扩展 Pose 和 Mask 到新画布
# =========================
def expand_to_canvas(image: Image.Image, new_size, paste_pos):
    """
    把原图尺寸的图像贴到新画布上
    """
    new_w, new_h = new_size
    canvas = Image.new(
        image.mode, (new_w, new_h), (0, 0, 0) if image.mode == "RGB" else 0
    )
    canvas.paste(image, paste_pos)
    return canvas


# =========================
# 5. Canny Control
# =========================
def make_canny_control(image: Image.Image, low=120, high=220, blur_sigma=0.8):
    np_img = np.array(image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    if blur_sigma and blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    edges = cv2.Canny(gray, threshold1=low, threshold2=high)
    edges_rgb = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_rgb, mode="RGB")


class FourStageOutpainter:
    """
    Encapsulate stage0~stage3 pipelines and model loading.

    - Stage implementations are private methods: `_stage0/_stage1/_stage2/_stage3`.
    - Utilities such as `calculate_wh` and `make_canny_control` remain module-level.
    - Constructor accepts optional `pose_getter` and `pipe_*` instances and supports lazy loading.
    """

    def __init__(
        self,
        cfg=None,
        device: str = "cuda",
        pose_getter: Optional[object] = None,
        pipe_inpaint: Optional[object] = None,
        pipe_canny: Optional[object] = None,
        pipe_openpose: Optional[object] = None,
        lazy_load: bool = True,
    ):
        self.cfg = cfg or get_default_config()
        self.device = device

        # model ids/paths from config
        self.animagine_model = self.cfg.animagine_xl
        self.base_model = self.cfg.diffusers_stable_diffusion_xl_inpainting_model
        self.ctl_model_canny = self.cfg.control_model_canny
        self.ctl_model_openpose = self.cfg.control_model_openpose
        self.vae_model = self.cfg.vae_model

        # optionally injected instances
        self.pose_getter = pose_getter
        self.pipe_inpaint = pipe_inpaint
        self.pipe_canny = pipe_canny
        self.pipe_openpose = pipe_openpose

        # lazy loading control
        self._models_loaded = False
        self._lazy = lazy_load

        if not self._lazy:
            self._lazy_load_models()

    def _lazy_load_models(self):
        if self._models_loaded:
            return

        print("🚀 正在准备模型路径和资源...")
        print("animagine_model:", self.animagine_model)
        print("base_model:", self.base_model)
        print("ctl_model_canny:", self.ctl_model_canny)
        print("ctl_model_openpose:", self.ctl_model_openpose)
        print("vae_model:", self.vae_model)

        print("\n🚀 正在加载模型...")

        if self.pose_getter is None:
            print("📌 加载 Pose 检测器...")
            self.pose_getter = RtmlibPoseGet()

        if self.pipe_inpaint is None:
            print("📌 加载 Base Inpaint Pipeline...")
            self.pipe_inpaint = load_inpaint_pipe(self.base_model, self.vae_model)

        if self.pipe_canny is None:
            print("📌 加载 Canny ControlNet...")
            self.pipe_canny = load_controlnet_inpaint_pipe(
                self.animagine_model, self.ctl_model_canny, self.vae_model
            )

        if self.pipe_openpose is None:
            print("📌 加载 OpenPose ControlNet...")
            self.pipe_openpose = load_controlnet_inpaint_pipe(
                self.animagine_model, self.ctl_model_openpose, self.vae_model
            )

        self._models_loaded = True
        print("✅ 模型加载完成\n")

    # -------------------------
    # Stage private methods
    # -------------------------
    def _stage0_extract_pose_and_mask(self, img_path, padding=20):
        image = load_image(img_path).convert("RGB")
        img_np = np.array(image)

        # ensure pose getter available
        if self.pose_getter is None:
            self._lazy_load_models()

        keypoints, scores = self.pose_getter.get_keypoints(img_np)

        if keypoints is None or len(keypoints) == 0:
            print("⚠️ 未检测到人物，返回空 mask")
            w, h = image.size
            return Image.new("RGB", (w, h), (0, 0, 0)), Image.new("L", (w, h), 0)

        h, w = img_np.shape[:2]

        pose_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        connections = [
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

        for kp in keypoints:
            for start_idx, end_idx in connections:
                if start_idx < len(kp) and end_idx < len(kp):
                    start = tuple(kp[start_idx].astype(int))
                    end = tuple(kp[end_idx].astype(int))
                    if scores[0][start_idx] > 0.3 and scores[0][end_idx] > 0.3:
                        cv2.line(pose_canvas, start, end, (255, 255, 255), 3)

        for kp in keypoints:
            for idx, point in enumerate(kp):
                if scores[0][idx] > 0.3:
                    cv2.circle(pose_canvas, tuple(point.astype(int)), 4, (0, 255, 0), -1)

        pose_image = Image.fromarray(pose_canvas)

        person_mask = np.zeros((h, w), dtype=np.uint8)

        for kp in keypoints:
            valid_points = []
            for idx, point in enumerate(kp):
                if scores[0][idx] > 0.3:
                    valid_points.append(point.astype(int))

            if len(valid_points) > 0:
                valid_points = np.array(valid_points)
                x_min = max(0, valid_points[:, 0].min() - padding)
                x_max = min(w, valid_points[:, 0].max() + padding)
                y_min = max(0, valid_points[:, 1].min() - padding)
                y_max = min(h, valid_points[:, 1].max() + padding)

                person_mask[y_min:y_max, x_min:x_max] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)

        person_mask_pil = Image.fromarray(person_mask)

        return pose_image, person_mask_pil

    def _stage1_base_inpaint(
        self,
        img_path,
        pose_image,
        person_mask,
        prompt,
        negative_prompt,
        feather_radius=12,
        guidance_scale=7.5,
        num_inference_steps=30,
        strength=0.98,
        debug_mask_path: Optional[str] = None,
    ):
        canvas, expand_mask, paste_pos = create_canvas_and_mask(
            img_path, feather_radius=feather_radius
        )

        new_w, new_h = canvas.size
        pose_canvas = expand_to_canvas(pose_image, (new_w, new_h), paste_pos)
        person_mask_canvas = expand_to_canvas(person_mask, (new_w, new_h), paste_pos)

        expand_np = np.array(expand_mask)
        person_np = np.array(person_mask_canvas)
        safe_mask_np = np.where(person_np > 128, 0, expand_np)
        safe_mask = Image.fromarray(safe_mask_np.astype(np.uint8))

        print(f"Stage1: Base Inpaint 扩展背景 {new_w}x{new_h}（保护人物）...")

        if debug_mask_path:
            os.makedirs(os.path.dirname(debug_mask_path) or ".", exist_ok=True)
            safe_mask.save(debug_mask_path)
            print(f"  调试：safe_mask 已保存到 {debug_mask_path}")

        if self.pipe_inpaint is None:
            self._lazy_load_models()

        result = self.pipe_inpaint(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canvas,
            mask_image=safe_mask,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            width=new_w,
            height=new_h,
        ).images[0]

        return result, expand_mask, person_mask_canvas, pose_canvas

    def _stage2_canny_background(
        self,
        base_image: Image.Image,
        expand_mask: Image.Image,
        person_mask: Image.Image,
        prompt,
        negative_prompt,
        controlnet_conditioning_scale=0.75,
        guidance_scale=6.0,
        num_inference_steps=35,
        strength=0.95,
        canny_low=120,
        canny_high=220,
        canny_blur_sigma=0.8,
        debug_canny_path=None,
        debug_background_mask_path: Optional[str] = None,
    ):
        w, h = base_image.size

        canny_control = make_canny_control(
            base_image, low=canny_low, high=canny_high, blur_sigma=canny_blur_sigma
        )

        if debug_canny_path:
            os.makedirs(os.path.dirname(debug_canny_path) or ".", exist_ok=True)
            canny_control.save(debug_canny_path)

        expand_np = np.array(expand_mask)
        person_np = np.array(person_mask)
        background_mask_np = np.where(person_np > 128, 0, expand_np)
        background_mask = Image.fromarray(background_mask_np.astype(np.uint8))

        if debug_background_mask_path:
            os.makedirs(os.path.dirname(debug_background_mask_path) or ".", exist_ok=True)
            background_mask.save(debug_background_mask_path)

        print(f"Stage2: Canny ControlNet 重构背景 {w}x{h}...")

        if self.pipe_canny is None:
            self._lazy_load_models()

        result = self.pipe_canny(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            mask_image=background_mask,
            control_image=canny_control,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            width=w,
            height=h,
        ).images[0]

        return result, background_mask

    def _stage3_openpose_refine_person(
        self,
        base_image: Image.Image,
        person_mask: Image.Image,
        pose_control: Image.Image,
        prompt,
        negative_prompt,
        controlnet_conditioning_scale=0.85,
        guidance_scale=6.5,
        num_inference_steps=40,
        strength=0.55,
        debug_pose_path=None,
        debug_person_mask_path: Optional[str] = None,
        background_mask=None,
    ):
        w, h = base_image.size

        if debug_pose_path:
            os.makedirs(os.path.dirname(debug_pose_path) or ".", exist_ok=True)
            pose_control.save(debug_pose_path)

        person_np = np.array(person_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        person_np = cv2.dilate(person_np, kernel, iterations=1)
        person_mask_expanded = Image.fromarray(person_np)

        if debug_person_mask_path:
            os.makedirs(os.path.dirname(debug_person_mask_path) or ".", exist_ok=True)
            person_mask_expanded.save(debug_person_mask_path)

        if background_mask is None:
            background_mask = Image.new("L", (w, h), 255)

        print(f"Stage3: OpenPose ControlNet 修正人物 {w}x{h}...")

        if self.pipe_openpose is None:
            self._lazy_load_models()

        result = self.pipe_openpose(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            mask_image=background_mask,
            control_image=pose_control,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
            width=w,
            height=h,
        ).images[0]

        return result

    # -------------------------
    # Orchestration
    # -------------------------
    def _execute_once(self, index: int, input_path: str, tmp_dir: str, save_path_template: str, debug: bool, prompt: str, negative_prompt: str):
        os.makedirs(tmp_dir, exist_ok=True)

        # Stage 0
        print("\n=== Stage 0: 提取 Pose 和人物 Mask ===")
        pose_image, person_mask = self._stage0_extract_pose_and_mask(input_path, padding=30)

        if debug:
            pose_path = os.path.join(tmp_dir, f"stage0_pose_{index}.png")
            mask_path = os.path.join(tmp_dir, f"stage0_mask_{index}.png")
            pose_image.save(pose_path)
            person_mask.save(mask_path)
            print(f"Pose 保存: {pose_path}")
            print(f"Mask 保存: {mask_path}")

        # Stage 1
        print("\n=== Stage 1: Base Inpaint 扩展背景 ===")
        stage1_img, expand_mask, person_mask_canvas, pose_canvas = self._stage1_base_inpaint(
            input_path,
            pose_image,
            person_mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            feather_radius=8,
            guidance_scale=7.5,
            num_inference_steps=40,
            strength=0.99,
            debug_mask_path=os.path.join(tmp_dir, f"stage1_safe_mask_{index}.png") if debug else None,
        )

        if debug:
            stage1_path = os.path.join(tmp_dir, f"stage1_inpaint_{index}.png")
            stage1_img.save(stage1_path)
            print(f"Stage1 保存: {stage1_path}")

        # Stage 2
        print("\n=== Stage 2: Canny 重构背景 ===")
        stage2_img, background_mask = self._stage2_canny_background(
            base_image=stage1_img,
            expand_mask=expand_mask,
            person_mask=person_mask_canvas,
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=0.65,
            guidance_scale=6.0,
            num_inference_steps=35,
            strength=0.82,
            canny_low=120,
            canny_high=220,
            canny_blur_sigma=0.8,
            debug_canny_path=os.path.join(tmp_dir, f"stage2_canny_{index}.png") if debug else None,
            debug_background_mask_path=os.path.join(tmp_dir, f"stage2_background_mask_{index}.png") if debug else None,
        )

        stage2_path = os.path.join(tmp_dir, f"stage2_result_{index}.png")
        stage2_img.save(stage2_path)
        print(f"Stage2 保存: {stage2_path}")

        # Stage 3
        print("\n=== Stage 3: OpenPose 修正人物 ===")
        final_img = self._stage3_openpose_refine_person(
            base_image=stage2_img,
            person_mask=person_mask_canvas,
            pose_control=pose_canvas,
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=0.80,
            guidance_scale=6.5,
            num_inference_steps=40,
            strength=0.20,
            debug_pose_path=os.path.join(tmp_dir, f"stage3_pose_{index}.png") if debug else None,
            debug_person_mask_path=os.path.join(tmp_dir, f"stage3_person_mask_{index}.png") if debug else None,
            background_mask=None,
        )

        save_path = save_path_template.format(index)
        final_img.save(save_path)
        print(f"\n✅ 完成第 {index} 次生成: {save_path}\n")

    def run(self, input_img_path, save_pic_path, prompt=None, negative_prompt=None, runs=5, debug=False):
        os.makedirs(save_pic_path, exist_ok=True)
        tmp_dir = os.path.join(save_pic_path, "tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        if not prompt:
            prompt = (
                "little girl, sitting in a forest, sitting on a table, many plants, "
                "soft lighting, clean and minimalistic composition, emphasizing her "
                "cuteness and purity, highly detailed, ultra-realistic, 8k resolution, "
                "studio lighting, vibrant colors"
            )

        if not negative_prompt:
            negative_prompt = (
                "lowres, (bad anatomy, bad hands:1.2, bad legs:1.2), text, error, "
                "missing fingers, extra digit, fewer digits, cropped, worst quality, "
                "low quality, normal quality, jpeg artifacts, signature, watermark, "
                "username, blurry"
            )

        save_path_template = os.path.join(save_pic_path, "res_{}.png")

        # Ensure models are available when first needed
        if not self._lazy:
            self._lazy_load_models()

        for i in range(1, runs + 1):
            print(f"\n{'=' * 60}")
            print(f"🎨 开始第 {i} 次四阶段扩图")
            print(f"{'=' * 60}")

            # load models on-demand right before executing
            if self._lazy and not self._models_loaded:
                self._lazy_load_models()

            self._execute_once(
                index=i,
                input_path=input_img_path,
                tmp_dir=tmp_dir,
                save_path_template=save_path_template,
                debug=debug,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )


def pic_expand(input_img_path, prompt, save_pic_path):
    """
    External interface: runs the four-stage outpaint and stores results under `save_pic_path`.
    """
    outpainter = FourStageOutpainter()
    debug = os.environ.get("PIC_EXPAND_DEBUG", "0") in {"1", "true", "True", "yes", "YES"}
    return outpainter.run(input_img_path, save_pic_path, prompt=prompt, runs=5, debug=debug)


if __name__ == "__main__":
    shutil.rmtree("./tmp", ignore_errors=True)
    input_img_path = "/home/ping/Downloads/todo_pic/111965731_p0_master1200.jpg"
    prompt = (
        "little girl, sitting in a forest, sitting on a table, many plants, "
        "soft lighting, clean and minimalistic composition, emphasizing her "
        "cuteness and purity, highly detailed, ultra-realistic, 8k resolution, "
        "studio lighting, vibrant colors"
    )
    save_pic_path = "./run"
    pic_expand(input_img_path=input_img_path, prompt=prompt, save_pic_path=save_pic_path)
    if input("是否删除临时文件？(y/n): ") == "y":
        shutil.rmtree("./tmp", ignore_errors=True)