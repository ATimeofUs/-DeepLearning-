from rtmlib import Wholebody

class RtmlibPoseGet:
    def __init__(self, device="cuda", backend="onnxruntime", st=True):
        self.wholebody = Wholebody(
            to_openpose=st,
            mode="performance",
            backend=backend,
            device=device,
        )

    def get_keypoints(self, img):
        keypoints, scores = self.wholebody(img)
        return keypoints, scores


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    pose_getter = RtmlibPoseGet(device="cuda", backend="onnxruntime", st=True)
    img = cv2.imread("/home/ping/src/my_python/run/wallhaven-vqoyql.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    keypoints, scores = pose_getter.get_keypoints(img_rgb)

    print("Keypoints shape:", keypoints.shape)
    print("Scores shape:", scores.shape)

    # 可视化关键点
    plt.imshow(img_rgb)
    for person_kp in keypoints:
        for x, y in person_kp:
            plt.scatter(x, y, c='red', s=10)
    plt.axis('off')
    plt.show()

    input("Press Enter to exit...")