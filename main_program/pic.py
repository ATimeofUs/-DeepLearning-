import os
from utils.my_diffusers.pic_expand import pic_expand


def main():
    pic_path = "/home/ping/Downloads/todo_pic/wallhaven-vqoyql.jpg"

    prompt = (
        "in a room, modern furniture, anime style, bookshelf, big sofa, coffee table, elegant lamp, few plants"
        "bright inside, nighttime outside, cozy atmosphere, correct perspective,"
        "soft lighting, clean and minimalistic composition, straight lines, correct perspective, architectural interior, clean geometry, sharp edges,"
        "highly detailed, ultra-realistic, 8k resolution"
    )

    print("提示词 (prompt):")
    print(prompt)

    save_path = "/home/ping/src/my_python/tmp"

    stage_params = {
        "stage0": {"padding": 24},
        "stage1": {"feather_radius": 4, "guidance_scale": 7.5, "num_inference_steps": 40, "strength": 0.98},
        "stage2": {"controlnet_conditioning_scale": 0.70, "guidance_scale": 7.5, "num_inference_steps": 30, "strength": 0.90},
        "stage3": {"controlnet_conditioning_scale": 0.70, "guidance_scale": 6.5, "num_inference_steps": 40, "strength": 0.75},
    }

    os.makedirs(save_path, exist_ok=True)
    pic_expand(
        input_img_path=pic_path,
        prompt=prompt,
        save_pic_path=save_path,
        debug=True,
        stage_params=stage_params,
    )


if __name__ == "__main__":
    main()
