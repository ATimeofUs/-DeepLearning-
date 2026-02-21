import cv2
import os
import subprocess
import shutil
from PIL import Image


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

def compose_half_and_half(pic_path_1, pic_path_2, output_path):
    img1 = cv2.imread(pic_path_1)
    img2 = cv2.imread(pic_path_2)

    # 确保两张图像尺寸相同
    
    if img1.shape != img2.shape:
        r1 = img1.shape[1] / img2.shape[1]
        r2 = img1.shape[0] / img2.shape[0]
        
        if abs(r1 - r2) > 0.01:
            raise ValueError("两张图像的宽高比差异过大，无法直接合成")
        
        if img1.shape[0] > img2.shape[0]:
            img1, img2 = img2, img1  # 交换，使得 img1 是较大的图像
        
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)
        

    h, w, c = img1.shape
    half_w = w // 2

    # 创建一个新的图像，左半部分来自 img1，右半部分来自 img2
    combined_img = cv2.hconcat([img1[:, :half_w], img2[:, half_w:]])

    # 保存合成后的图像
    cv2.imwrite(output_path, combined_img)
    

if __name__ == "__main__":
    dir = "/home/ping/Pictures/background/todo_pic/"

    path_list = os.listdir(dir)
    
    for path in path_list:
        if path.endswith(".jpg") or path.endswith(".png"):
            img = Image.open(os.path.join(dir, path))
            w, h = img.size
            
            if w > 2560 or h > 1600:
                remove_path = os.path.join(dir, path)
                print(f"Removing {remove_path} (size={w}x{h})")
                os.remove(remove_path)
                # ratio = w / h
                
                # if w > 2560:
                #     new_w = 2560
                #     new_h = int(new_w / ratio)
                # else:
                #     new_h = 1600
                #     new_w = int(new_h * ratio)
                
                # img = img.resize((new_w, new_h), Image.ANTIALIAS)
                # img.save(os.path.join(dir, "new_" + path))