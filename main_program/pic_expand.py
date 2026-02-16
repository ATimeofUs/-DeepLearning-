import os
import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter
from simple_lama_inpainting import SimpleLama

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    AutoencoderKL,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.utils import load_image
from utils.config import get_default_config


# =========================
# 1. 计算 16:10 画布，并对齐到 64（向上取整）
# =========================
def _ceil_to(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def calculate_wh(h: int, w: int, ratio: float = 10 / 16, align: int = 64):
    """
    目标：得到 >= 原图的 16:10 画布尺寸，并对齐到 align 的倍数（默认 64）。
    ratio = H/W
    """
    # 先按比例扩展到 >= 原图
    if h / w < ratio:
        # 原图太“矮”(相对宽)，需要增加高度
        new_w = w
        new_h = int(round(w * ratio))
    else:
        # 原图太“高”，需要增加宽度
        new_h = h
        new_w = int(round(h / ratio))

    # 再做对齐（向上取整），保证不小于上述值
    new_w = _ceil_to(new_w, align)
    new_h = _ceil_to(new_h, align)
    return new_h, new_w


# =========================
# 2. 创建 Canvas + Mask
#    mask: 白(255)=重绘区域, 黑(0)=保留区域
#    feather_radius 控制羽化过渡宽度
# =========================
def create_canvas_and_mask(img_path, feather_radius=12):
    image = load_image(img_path).convert("RGB")
    w, h = image.size

    new_h, new_w = calculate_wh(h, w, ratio=10 / 16, align=64)
    p_x = (new_w - w) // 2
    p_y = (new_h - h) // 2

    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    canvas.paste(image, (p_x, p_y))

    # 先全白（全重绘），再把原图区域涂黑（保留）
    mask = Image.new("L", (new_w, new_h), 255)
    keep = Image.new("L", (w, h), 0)
    mask.paste(keep, (p_x, p_y))

    # 羽化：让边缘融合（这会在原图边缘产生灰度过渡）
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return canvas, mask


# =========================
# 3. 可选：LaMa 预填充底图（如果安装了 simple_lama_inpainting）
# =========================
def lama_fill(canvas: Image.Image, mask: Image.Image):
    lama = SimpleLama()
    filled = lama(canvas, mask)
    return filled.convert("RGB")


# =========================
# 4. 生成 Canny 控制图（适合二次元/线稿约束）
# =========================
def make_canny_control(image: Image.Image, low=120, high=220, blur_sigma=0.8):
    np_img = np.array(image)  # RGB HWC
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    if blur_sigma and blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    edges = cv2.Canny(gray, threshold1=low, threshold2=high)
    edges_rgb = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_rgb, mode="RGB")


# =========================
# 5. 加载 Pipelines
# =========================
def load_inpaint_pipe(base_moodel, vae_model):
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_moodel,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe.enable_sequential_cpu_offload()
    return pipe


def load_controlnet_inpaint_pipe(base_moodel, control_model, vae_model):
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
        base_moodel,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe.enable_sequential_cpu_offload()
    return pipe


# =========================
# 6. Stage1（可选）：无 ControlNet 的预扩图（可用 LaMa 作为底图）
# =========================
def stage1_pre_outpaint(
    pipe_inpaint,
    img_path,
    prompt,
    negative_prompt,
    use_lama=True,
    feather_radius=12,
    guidance_scale=7.5,
    num_inference_steps=30,
    strength=0.85,
):
    canvas, mask = create_canvas_and_mask(img_path, feather_radius=feather_radius)

    if use_lama:
        print("Stage1: LaMa 填充中...")
        base_image = lama_fill(canvas, mask)
    else:
        base_image = canvas

    w, h = canvas.size
    print(f"Stage1: SDXL inpaint 推理 {w}x{h} ...")

    result = pipe_inpaint(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        width=w,
        height=h,
    ).images[0]

    return result, mask


# =========================
# 7. Stage2：Canny ControlNet 扩图（只重绘扩展区域）
# =========================
def stage2_outpaint_canny(
    pipe_control,
    base_image: Image.Image,
    mask: Image.Image,
    prompt,
    negative_prompt,
    controlnet_conditioning_scale=0.75,
    guidance_scale=6.0,
    num_inference_steps=35,
    strength=0.92,
    canny_low=120,
    canny_high=220,
    canny_blur_sigma=0.8,
    debug_canny_path=None,
    redraw_all=True,
):
    w, h = base_image.size

    canny_control = make_canny_control(
        base_image, low=canny_low, high=canny_high, blur_sigma=canny_blur_sigma
    )

    if redraw_all:
        mask = Image.new("L", (w, h), 255)

    if debug_canny_path:
        os.makedirs(os.path.dirname(debug_canny_path) or ".", exist_ok=True)
        canny_control.save(debug_canny_path)

    print(f"Stage2: ControlNet Canny 推理 {w}x{h} ...")
    result = pipe_control(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        mask_image=mask,
        control_image=canny_control,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        width=w,
        height=h,
    ).images[0]

    return result


def run_outpaint(
    index: int,
    pipe_inpaint,
    pipe_control,
    input_path: str,
    tmp_dir: str,
    save_path_template: str = "./run/res_{}.png",
):
    os.makedirs(tmp_dir, exist_ok=True)

    prompt = "1girl, solo, lying down on a red and white bed with plush pillows and soft sheets, dark floor beneath, pink hair styled with red and white ribbons tied into bows, fluffy fox ears perked up"
    negative_prompt = "lowres, bad anatomy, worst quality, blurry, photorealistic, 3d render, three legs"

    use_stage1 = True
    use_lama = True

    if use_stage1:
        stage1_img, mask = stage1_pre_outpaint(
            pipe_inpaint,
            input_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            use_lama=use_lama,
            feather_radius=12,
            guidance_scale=7.5,
            num_inference_steps=30,
            strength=0.85,
        )
        stage1_path = os.path.join(tmp_dir, f"stage1_{index}.png")
        stage1_img.save(stage1_path)
        print(f"Stage1 保存: {stage1_path}")
        base_image = stage1_img
    else:
        base_image, mask = create_canvas_and_mask(input_path, feather_radius=12)

    final = stage2_outpaint_canny(
        pipe_control,
        base_image=base_image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.55,
        guidance_scale=6.0,
        num_inference_steps=35,
        strength=0.68,
        canny_low=120,
        canny_high=220,
        canny_blur_sigma=0.8,
        debug_canny_path=os.path.join(tmp_dir, f"debug_canny_{index}.png"),
    )

    save_path = save_path_template.format(index)
    final.save(save_path)
    print(f"完成第 {index} 次生成: {save_path}")


def main():
    # load configuration (can be overridden by env vars)
    cfg = get_default_config()

    animagine_model = cfg.animagine_xl
    base_moodel = cfg.diffusers_stable_diffusion_xl_inpainting_model
    
    ctl_model_canny = cfg.control_model_canny
    ctl_model_openpose = cfg.control_model_openpose
    
    vae_model = cfg.vae_model
    input_img_path = cfg.input_path
    
    print("正在加载模型...")

    # Stage1: 用 base_moodel 做基础扩图（先把画布和大结构补齐）
    pipe_inpaint = load_inpaint_pipe(base_moodel, vae_model)

    # Stage2: 从 Stage1 结果提取 Canny，并用 animagine_model 进行重绘风格化
    pipe_control = load_controlnet_inpaint_pipe(
        animagine_model, ctl_model_canny, vae_model
    )

    for i in range(1, 6):
        print(f"\n>>> 开始第 {i} 次运行...")
        run_outpaint(
            i, 
            pipe_inpaint, 
            pipe_control, 
            input_img_path, 
            tmp_dir="./tmp", 
            save_path_template= f"./run/res_{i}.png"
        )


if __name__ == "__main__":
    main()
