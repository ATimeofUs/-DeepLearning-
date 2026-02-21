import os
import subprocess
import shutil
import torch

from utils.my_diffusers.pic_expand import (
    TwoStageOutpaint,
    ThreeStageOutpaint,
    create_canvas_and_mask,
)

from PIL import Image


def process_two_stage(pic_path = None, tmp_path = None, prepared_image_path = None):
    """
    使用 TwoStageOutpaint 进行扩图 (跳过 Stage1 Base Inpaint)。
    我们需要手动准备 Stage 0 的扩展图 (prepared_image)。
    这里演示使用 create_canvas_and_mask 生成一个 smart_fill 的扩展图作为 prepared_image。
    """

    if tmp_path is None or prepared_image_path is None:
        raise ValueError("tmp_path 和 prepared_image_path 参数必须提供")

    prompt = (
        "(sea), (sky), (water), few clouds, sunlight, [sand], 1girl, standing pose, short hair, no person, few  plants, "
        "a central focal point, clean edges, harmonious atmosphere, modern anime, balanced layout, fantasy art, masterpiece, best quality, very aesthetic, correct perspective, 8K"
    )

    negative_prompt = "(worst quality, low quality:1.2), (bad), error, (blurry), text, cropped, artifact, watermark, signature, (username:1.2), realistic, photo, photorealistic, 3d, cgi, bad hands, bad anatomy, disfigured, deformed, extra limbs, close up, b&w, weird colors,"

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

    if pic_path is None:
        # TODO 截断左边25%做为输入
        original_image = Image.open(prepared_image_path)
        w, h = original_image.size
        left_crop = int(w * 0.25)
        input_image = original_image.crop((left_crop, 0, w, h))
        input_image_path = os.path.join(tmp_path, "stage0_input.png")
        input_image.save(input_image_path)
        pic_path = input_image_path

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
        "(flat sea), 1gril, standing in sea, beach, sky, water, sideways view, large scene, less hair, no person, "
        "a central focal point, clean edges, harmonious atmosphere, modern anime, balanced layout, fantasy art, masterpiece, best quality, very aesthetic, correct perspective, 8K"
    )

    negative_prompt = "wrong arms, wrong legs, wrong hair, (worst quality, low quality:1.2), (bad), error, (blurry), text, cropped, artifact, watermark, signature, (username:1.2), realistic, photo, photorealistic, 3d, cgi, bad hands, bad anatomy, disfigured, deformed, extra limbs, close up, b&w, weird colors,"

    # mask_mode 参数选择使用的 mask（0 = expand_mask, 1=全白 mask）
    # model_type 参数选择使用的模型（0=BASE 模型，1=ANIME 模型）
    # image_source 参数选择 ControlNet 输入图像来源（"canvas" 或 "current"）
    # feather_radius 参数控制边缘羽化程度，数值越大羽化越明显，建议在 0-5 之间调整
    # guidance_scale 和 controlnet_conditioning_scale 参数控制生成质量和对 ControlNet 的依赖程度

    stage_params = {
        "stage1": {
            "feather_radius": 1,
            "guidance_scale": 7.0, 
            "num_inference_steps": 35,
            "strength": 0.99, # 保留一定的原图细节，避免过度修改
            "model_type": 0,
        },
        "stage2": {
            "controlnet_conditioning_scale": [0.70, 0.70],
            "guidance_scale": 2.0,
            "num_inference_steps": 40,
            "strength": 0.98, # 保留更多细节，避免过度修改
            "model_type": 0,
            "image_source": "current",
            "mask_mode": 0,
        },
        "stage3": {
            "controlnet_conditioning_scale": [0.60, 0.85],
            "guidance_scale": 10.0,
            "num_inference_steps": 35,
            "strength": 0.93, # 适当增加修改程度，提升细节质量
            "model_type": 1,
            "image_source": "current",
            "mask_mode": 0,
        },
    }

    shutil.rmtree(tmp_path, ignore_errors=True)  # 清理旧的临时目录
    os.makedirs(tmp_path, exist_ok=True)

    # 创建 ThreeStageOutpaint 实例并执行
    outpainter = ThreeStageOutpaint(
        device="cuda", lazy_load=True, stage_params=stage_params
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


if __name__ == "__main__":
    input_path = "/home/ping/Pictures/background/todo_pic/new_wallhaven-gwzomd.jpg"
    
    for i in range(6):
        print(f"Processing image {i+1}/6: {input_path}")
        tmp_dir = f"tmp/res_{i}"
        process_three_stage(input_path, tmp_dir)
        
        torch.cuda.empty_cache()  # 清理 GPU 内存