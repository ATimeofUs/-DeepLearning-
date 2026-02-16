import os
import numpy as np
import cv2
import torch
import shutil
from PIL import Image, ImageFilter
from utils.deep_learning import RtmlibPoseGet

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    AutoencoderKL,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.utils import load_image
from utils.config import get_default_config


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
def create_canvas_and_mask(img_path, feather_radius=12):
    image = load_image(img_path).convert("RGB")
    w, h = image.size

    new_h, new_w = calculate_wh(h, w, ratio=10 / 16, align=64)
    p_x = (new_w - w) // 2
    p_y = (new_h - h) // 2

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
# 3. Stage 0：提取 Pose + 人物 Mask
# =========================
def stage0_extract_pose_and_mask(img_path, pose_getter, padding=20):
    """
    提取人物骨架图 + 人物区域 mask
    返回：
        - pose_image: 骨架图 (PIL RGB)
        - person_mask: 人物区域 mask (PIL L)，白色=人物
    """
    image = load_image(img_path).convert("RGB")
    img_np = np.array(image)

    # 获取关键点
    keypoints, scores = pose_getter.get_keypoints(img_np)

    if keypoints is None or len(keypoints) == 0:
        print("⚠️ 未检测到人物，返回空 mask")
        w, h = image.size
        return Image.new("RGB", (w, h), (0, 0, 0)), Image.new("L", (w, h), 0)

    h, w = img_np.shape[:2]

    # 1. 绘制骨架图
    pose_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # OpenPose 连接关系（COCO 17点）
    connections = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # 头部
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # 上半身
        (5, 11),
        (6, 12),
        (11, 12),  # 躯干
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # 下半身
    ]

    # 画骨架线
    for kp in keypoints:
        for start_idx, end_idx in connections:
            if start_idx < len(kp) and end_idx < len(kp):
                start = tuple(kp[start_idx].astype(int))
                end = tuple(kp[end_idx].astype(int))
                if scores[0][start_idx] > 0.3 and scores[0][end_idx] > 0.3:
                    cv2.line(pose_canvas, start, end, (255, 255, 255), 3)

    # 画关键点
    for kp in keypoints:
        for idx, point in enumerate(kp):
            if scores[0][idx] > 0.3:
                cv2.circle(pose_canvas, tuple(point.astype(int)), 4, (0, 255, 0), -1)

    pose_image = Image.fromarray(pose_canvas)

    # 2. 创建人物 mask
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

    # 形态学闭运算，填补人物区域空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)

    person_mask_pil = Image.fromarray(person_mask)

    return pose_image, person_mask_pil



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


# =========================
# 6. 加载 Pipelines
# =========================
def load_inpaint_pipe(base_model, vae_model):
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.enable_sequential_cpu_offload()
    return pipe


def load_controlnet_inpaint_pipe(base_model, control_model, vae_model):
    controlnet = ControlNetModel.from_pretrained(
        control_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.enable_sequential_cpu_offload()
    return pipe


# =========================
# 7. Stage 1：base_model inpaint 扩展背景
# =========================
def stage1_base_inpaint(
    pipe_inpaint,
    img_path,
    pose_image,
    person_mask,
    prompt,
    negative_prompt,
    feather_radius=12,
    guidance_scale=7.5,
    num_inference_steps=30,
    strength=1.0,
):
    """
    用 base_model 的 inpaint 扩展背景
    保护人物区域不被重绘
    """
    canvas, expand_mask, paste_pos = create_canvas_and_mask(
        img_path, feather_radius=feather_radius
    )

    # 扩展 pose 和 mask 到新画布
    new_w, new_h = canvas.size
    pose_canvas = expand_to_canvas(pose_image, (new_w, new_h), paste_pos, color=255)
    person_mask_canvas = expand_to_canvas(person_mask, (new_w, new_h), paste_pos, color=0)

    # 合并 mask：只重绘扩展区域，不碰人物
    expand_np = np.array(expand_mask)
    person_np = np.array(person_mask_canvas)
    safe_mask_np = np.where(person_np > 128, 0, expand_np)
    safe_mask = Image.fromarray(safe_mask_np.astype(np.uint8))

    print(f"Stage1: Base Inpaint 扩展背景 {new_w}x{new_h}（保护人物）...")
    result = pipe_inpaint(
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


# =========================
# 8. Stage 2：Canny 重构背景
# =========================
def stage2_canny_background(
    pipe_control,
    base_image: Image.Image,
    expand_mask: Image.Image,
    person_mask: Image.Image,
    prompt,
    negative_prompt,
    controlnet_conditioning_scale=0.75,
    guidance_scale=6.0,
    num_inference_steps=35,
    strength=0.85,
    canny_low=120,
    canny_high=220,
    canny_blur_sigma=0.8,
    debug_canny_path=None,
):
    w, h = base_image.size

    # 生成 Canny 控制图
    canny_control = make_canny_control(
        base_image, low=canny_low, high=canny_high, blur_sigma=canny_blur_sigma
    )

    if debug_canny_path:
        os.makedirs(os.path.dirname(debug_canny_path) or ".", exist_ok=True)
        canny_control.save(debug_canny_path)

    # mask: 只重绘扩展区域（不碰人物）
    expand_np = np.array(expand_mask)
    person_np = np.array(person_mask)
    background_mask_np = np.where(person_np > 128, 0, expand_np)
    background_mask = Image.fromarray(background_mask_np.astype(np.uint8))

    print(f"Stage2: Canny ControlNet 重构背景 {w}x{h}...")
    result = pipe_control(
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

    return result


# =========================
# 9. Stage 3：OpenPose 修正人物
# =========================
def stage3_openpose_refine_person(
    pipe_control,
    base_image: Image.Image,
    person_mask: Image.Image,
    pose_control: Image.Image,
    prompt,
    negative_prompt,
    controlnet_conditioning_scale=0.85,
    guidance_scale=6.5,
    num_inference_steps=40,
    strength=0.75,
    debug_pose_path=None,
):
    w, h = base_image.size

    if debug_pose_path:
        os.makedirs(os.path.dirname(debug_pose_path) or ".", exist_ok=True)
        pose_control.save(debug_pose_path)

    # 扩展 person_mask 一点，确保覆盖完整
    person_np = np.array(person_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    person_np = cv2.dilate(person_np, kernel, iterations=1)
    person_mask_expanded = Image.fromarray(person_np)

    print(f"Stage3: OpenPose ControlNet 修正人物 {w}x{h}...")
    result = pipe_control(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        mask_image=person_mask_expanded,
        control_image=pose_control,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        width=w,
        height=h,
    ).images[0]

    return result


# =========================
# 10. 完整四阶段流程
# =========================
def run_four_stage_outpaint(
    index: int,
    pose_getter,
    pipe_inpaint,
    pipe_canny,
    pipe_openpose,
    input_path: str,
    tmp_dir: str,
    save_path_template: str = "./run/res_{}.png",
):
    os.makedirs(tmp_dir, exist_ok=True)

    prompt = "1girl, solo, lying down on a red and white bed with plush pillows and soft sheets, peaceful emotion, dark floor beneath, pink hair styled with red and white ribbons tied into bows, fluffy fox ears perked up, anime style, high quality"

    negative_prompt = "lowres, bad anatomy, worst quality, blurry, photorealistic, 3d render, extra limbs, bad hands, malformed limbs"

    # ========== Stage 0 ==========
    print("\n=== Stage 0: 提取 Pose 和人物 Mask ===")
    pose_image, person_mask = stage0_extract_pose_and_mask(
        input_path, pose_getter, padding=30
    )

    pose_path = os.path.join(tmp_dir, f"stage0_pose_{index}.png")
    mask_path = os.path.join(tmp_dir, f"stage0_mask_{index}.png")
    pose_image.save(pose_path)
    person_mask.save(mask_path)
    print(f"Pose 保存: {pose_path}")
    print(f"Mask 保存: {mask_path}")

    # ========== Stage 1 ==========
    print("\n=== Stage 1: Base Inpaint 扩展背景 ===")
    stage1_img, expand_mask, person_mask_canvas, pose_canvas = stage1_base_inpaint(
        pipe_inpaint,
        input_path,
        pose_image,
        person_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        feather_radius=12,
        guidance_scale=7.5,
        num_inference_steps=30,
        strength=0.85,
    )

    stage1_path = os.path.join(tmp_dir, f"stage1_inpaint_{index}.png")
    stage1_img.save(stage1_path)
    print(f"Stage1 保存: {stage1_path}")

    # ========== Stage 2 ==========
    print("\n=== Stage 2: Canny 重构背景 ===")
    stage2_img = stage2_canny_background(
        pipe_canny,
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
        debug_canny_path=os.path.join(tmp_dir, f"stage2_canny_{index}.png"),
    )

    stage2_path = os.path.join(tmp_dir, f"stage2_result_{index}.png")
    stage2_img.save(stage2_path)
    print(f"Stage2 保存: {stage2_path}")

    # ========== Stage 3 ==========
    print("\n=== Stage 3: OpenPose 修正人物 ===")
    final_img = stage3_openpose_refine_person(
        pipe_openpose,
        base_image=stage2_img,
        person_mask=person_mask_canvas,
        pose_control=pose_canvas,
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.80,
        guidance_scale=6.5,
        num_inference_steps=40,
        strength=0.50,
        debug_pose_path=os.path.join(tmp_dir, f"stage3_pose_{index}.png"),
    )

    save_path = save_path_template.format(index)
    final_img.save(save_path)
    print(f"\n✅ 完成第 {index} 次生成: {save_path}\n")


# =========================
# 11. 主函数
# =========================
def main():
    cfg = get_default_config()

    animagine_model = cfg.animagine_xl
    base_model = cfg.diffusers_stable_diffusion_xl_inpainting_model
    ctl_model_canny = cfg.control_model_canny
    ctl_model_openpose = cfg.control_model_openpose
    vae_model = cfg.vae_model
    input_img_path = "/home/ping/src/my_python/run/hongxue.jpg"

    print("🚀 正在准备模型路径和资源...")
    print("animagine_model:", animagine_model)
    print("base_model:", base_model)
    print("ctl_model_canny:", ctl_model_canny)
    print("ctl_model_openpose:", ctl_model_openpose)
    print("vae_model:", vae_model)
    print("input_img_path:", input_img_path)

    print("\n🚀 正在加载模型...")

    # Pose 检测器
    print("📌 加载 Pose 检测器...")
    pose_getter = RtmlibPoseGet()

    # Stage 1: Base Inpaint
    print("📌 加载 Base Inpaint Pipeline...")
    pipe_inpaint = load_inpaint_pipe(base_model, vae_model)

    # Stage 2: Canny ControlNet
    print("📌 加载 Canny ControlNet...")
    pipe_canny = load_controlnet_inpaint_pipe(
        animagine_model, ctl_model_canny, vae_model
    )

    # Stage 3: OpenPose ControlNet
    print("📌 加载 OpenPose ControlNet...")
    pipe_openpose = load_controlnet_inpaint_pipe(
        animagine_model, ctl_model_openpose, vae_model
    )

    print("✅ 模型加载完成\n")

    # 运行 5 次生成
    for i in range(1, 6):
        print(f"\n{'=' * 60}")
        print(f"🎨 开始第 {i} 次四阶段扩图")
        print(f"{'=' * 60}")

        run_four_stage_outpaint(
            i,
            pose_getter,
            pipe_inpaint,
            pipe_canny,
            pipe_openpose,
            input_img_path,
            tmp_dir="./tmp",
            save_path_template=f"./run/res_{i}.png",
        )


if __name__ == "__main__":
    shutil.rmtree("./tmp", ignore_errors=True)
    main()

    if input("是否删除临时文件？(y/n): ") == "y":
        shutil.rmtree("./tmp", ignore_errors=True)
