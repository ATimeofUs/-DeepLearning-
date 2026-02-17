import os
from PIL import Image
from utils.deep_learning.pic_expand import pic_expand

def main():
    pic_path = "/home/ping/Downloads/todo_pic/wallhaven-vqoyql.jpg"

    prompt = (
        "many furnitures around, sitting on a sofa, in a room, outside ths window is city, in night, warn lighting"
        "soft lighting, clean and minimalistic composition, emphasizing her"
        "cuteness and purity, highly detailed, ultra-realistic, 8k resolution, "
        "vibrant colors"
    )

    save_path = "/home/ping/Downloads/todo_pic/res"

    os.makedirs(save_path, exist_ok=True)
    pic_expand(input_img_path=pic_path, prompt=prompt, save_pic_path=save_path)

if __name__ == "__main__":
    main()
