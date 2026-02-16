import cv2
from rtmlib import Wholebody, draw_skeleton
import numpy as np

img = cv2.imread("hongxue.jpg")

device = "cuda"  # cpu, cuda, mps
backend = "onnxruntime"  # opencv, onnxruntime, openvino
st = True

wholebody = Wholebody(
    to_openpose=st,  # 是否转换为 OpenPose 格式的骨架（25 关键点）。如果为 False，则使用原始的 133 关键点。
    mode="performance",  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
    backend=backend,
    device=device,
)

keypoints, scores = wholebody(img)

# img为一个黑色图像，大小与输入图像相同，用于绘制骨架。
img_black = np.zeros(img.shape, dtype=np.uint8)

img = draw_skeleton(img_black, keypoints, scores, kpt_thr=0.5, openpose_skeleton=st)

cv2.imwrite("./tmp/result.jpg", img)
