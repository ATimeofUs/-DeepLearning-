import os
import subprocess
import shutil
from utils.my_diffusers.pic_expand import (
    TwoStageOutpaint, 
    ThreeStageOutpaint, 
    create_canvas_and_mask
)

from PIL import Image

def process_two_stage(pic_path, tmp_path, prepared_image_path):
    """
    使用 TwoStageOutpaint 进行扩图 (跳过 Stage1 Base Inpaint)。
    我们需要手动准备 Stage 0 的扩展图 (prepared_image)。
    这里演示使用 create_canvas_and_mask 生成一个 smart_fill 的扩展图作为 prepared_image。
    """

    prompt = (
        "(sea), (sky), (water), few clouds, sunlight, [sand], 1girl, standing pose, short hair, no person, few  plants, "
        "a central focal point, clean edges, harmonious atmosphere, modern anime, balanced layout, fantasy art, masterpiece, best quality, very aesthetic, correct perspective, 8K"
    )

    negative_prompt = (
        "(worst quality, low quality:1.2), (bad), error, (blurry), text, cropped, artifact, watermark, signature, (username:1.2), realistic, photo, photorealistic, 3d, cgi, bad hands, bad anatomy, disfigured, deformed, extra limbs, close up, b&w, weird colors,"
    )

    stage_params = {
        "stage2": {
            "controlnet_conditioning_scale": [0.70, 0.70],
            "guidance_scale": 8.0,
            "num_inference_steps": 40,
            "strength": 0.95,
            "model_type": 0,  
            "image_source": "current", 
            "mask_mode": 0,  
        },
        "stage3": {
            "controlnet_conditioning_scale": [0.60, 0.85],
            "guidance_scale": 10.0,
            "num_inference_steps": 35,
            "strength": 0.90,
            "model_type": 1, 
            "image_source": "current", 
            "mask_mode": 0,  
        },
    }

    shutil.rmtree(tmp_path, ignore_errors=True)  # 清理旧的临时目录
    os.makedirs(tmp_path, exist_ok=True)

    prepared_image = Image.open(prepared_image_path)
    outpainter = TwoStageOutpaint(stage_params=stage_params)

    context = outpainter.execute_once(
        input_path=pic_path,
        prepared_image=prepared_image,
        tmp_dir=tmp_path,  # 开启 debug 保存中间结果
        debug=True,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )

    final_path = os.path.join(tmp_path, "final_two_stage_res.png")
    context.current_image.save(final_path)
    print(f"🎉 Two-Stage processing done! Saved to: {final_path}")

def process_three_stage(pic_path, tmp_path):
    """
    三阶段 Outpaint 处理函数
    
    :param pic_path: 输入图像路径
    :param tmp_path: 临时输出目录
    """
    prompt = (
        "(sea), (full sky), (water), few clouds, sunlight, [sand], 1girl, standing pose, tilt view to left, sideway glance, short hair, no person, few  plants, "
        "a central focal point, clean edges, harmonious atmosphere, modern anime, balanced layout, fantasy art, masterpiece, best quality, very aesthetic, correct perspective, 8K"
    )

    negative_prompt = (
        "(worst quality, low quality:1.2), (bad), error, (blurry), text, cropped, artifact, watermark, signature, (username:1.2), realistic, photo, photorealistic, 3d, cgi, bad hands, bad anatomy, disfigured, deformed, extra limbs, close up, b&w, weird colors,"
    )

    # mask_mode 参数选择使用的 mask（0 = expand_mask, 1=全白 mask）
    # model_type 参数选择使用的模型（0=BASE 模型，1=ANIME 模型）
    # image_source 参数选择 ControlNet 输入图像来源（"canvas" 或 "current"）
    
    stage_params = {
        "stage1": {
            "feather_radius": 1,
            "guidance_scale": 7.0,
            "num_inference_steps": 35,
            "strength": 0.999,
            "model_type": 0,  
        },
        "stage2": {
            "controlnet_conditioning_scale": [0.70, 0.70],
            "guidance_scale": 2.0,
            "num_inference_steps": 40,
            "strength": 0.98,
            "model_type": 0,  
            "image_source": "current", 
            "mask_mode": 0,  
        },
        "stage3": {
            "controlnet_conditioning_scale": [0.60, 0.85],
            "guidance_scale": 10.0,
            "num_inference_steps": 35,
            "strength": 0.90,
            "model_type": 1, 
            "image_source": "current", 
            "mask_mode": 0,  
        },
    }

    shutil.rmtree(tmp_path, ignore_errors=True)  # 清理旧的临时目录
    os.makedirs(tmp_path, exist_ok=True)

    # 创建 ThreeStageOutpaint 实例并执行
    outpainter = ThreeStageOutpaint(
        device="cuda",  
        lazy_load=True,  
        stage_params=stage_params
    )
    
    context = outpainter.execute_once(
        input_path=pic_path,
        tmp_dir=tmp_path,
        debug=True,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )

    final_path = os.path.join(tmp_path, "final_three_stage_res.png")
    context.current_image.save(final_path)
    print(f"\n🎉 Three-Stage processing done! Saved to: {final_path}")
    
    del outpainter  # 显式删除实例以释放资源
    
    return context

def pic_x4(pic_path, new_pic_path):
    cmd = [
        "realesrgan-ncnn-vulkan",
        "-i",
        pic_path,
        "-o",
        new_pic_path,
        "-s",
        "4",
        "-n",
        "realesrgan-x4plus-anime",
        "-t",
        "0",  # 这里的 0 通常指 tile size 自动，或者是线程数，取决于版本
        "-g",
        "0",  # 指定第 0 块 GPU (你的 5070)
        "-f",
        "jpg",
    ]

    subprocess.run(cmd, check=True)

def main():
    main_dir = "/home/ping/Downloads/todo_pic"

    # cache directory to store intermediate upscaled images; filenames kept the same
    cache_dir = os.path.join(main_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for fname in os.listdir(main_dir):
        fpath = os.path.join(main_dir, fname)
        if not os.path.isfile(fpath):
            continue
        _, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            continue

        try:
            im = Image.open(fpath)
        except Exception as e:
            print(f"Skipping {fpath}: cannot open ({e})")
            continue

        w, h = im.size
        print(f"Processing {fname} (size={w}x{h})")

        # If either dimension is smaller than target, upscale 4x three times
        target_w, target_h = 2560, 1600
        current_input = fpath
        if w < target_w or h < target_h:
            for i in range(3):
                out_path = os.path.join(cache_dir, fname)  # keep same filename in cache
                print(f"  Upscaling pass {i+1}/3: {current_input} -> {out_path}")
                try:
                    pic_x4(current_input, out_path)
                except subprocess.CalledProcessError as e:
                    print(f"  Upscale failed on pass {i+1} for {fname}: {e}")
                    break
                current_input = out_path

        # Prepare per-image output directory; generated images for this
        # original image will be saved here. Upscaled intermediate files
        # remain in cache_dir (same filename).
        base_name = os.path.splitext(fname)[0]
        per_image_dir = os.path.join(main_dir, base_name)
        os.makedirs(per_image_dir, exist_ok=True)

        # Run three-stage pipeline with debug=False and save into per-image dir
        try:
            process_three_stage(current_input, per_image_dir)
        except Exception as e:
            print(f"Processing failed for {fname}: {e}")
            continue

if __name__ == "__main__":
    input_image_path = "/home/ping/Pictures/background/todo_pic/wallhaven-gwzomd.jpg"
    tmp_output_dir = "tmp"
    
    process_three_stage(input_image_path, tmp_output_dir)

