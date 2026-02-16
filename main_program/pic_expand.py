import os
from PIL import Image
from utils.deep_learning.pic_expand import pic_expand

def main():
    todo_file = "/home/ping/Downloads/todo_pic/"
    pic_name = os.listdir(todo_file)

    for pic in pic_name:
        pic_path = os.path.join(todo_file, pic)

        save_path = os.path.join(todo_file, pic.split(".")[0])
        os.makedirs(save_path, exist_ok=True)
        
        prompt = (
            "little girl, sitting in a forest, sitting on a table, many plants, "
            "soft lighting, clean and minimalistic composition, emphasizing her "
            "cuteness and purity, highly detailed, ultra-realistic, 8k resolution, "
            "studio lighting, vibrant colors"
        )

        pic_expand(input_img_path=pic_path, prompt=prompt, save_pic_path=save_path)


if __name__ == "__main__":
    main()
