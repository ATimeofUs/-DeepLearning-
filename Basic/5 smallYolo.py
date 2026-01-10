import torch 
from ultralytics import YOLO  # type: ignore
from ultralytics.nn import modules
from ultralytics.nn.tasks import parse_model
# 导入 YOLO 内部存储所有模块的字典

class args():
    model_dir = "./my.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"


def main():
    model = YOLO(args.model_dir).load('yolov8n.pt')

    results = model.predict(
        data=args.model_dir,
        source="https://ultralytics.com/images/bus.jpg", 
        conf=0.25, 
        save=True,
        device=args.device
    )

    for res in results:
        res.show()  

    breakpoint()

if __name__ == "__main__":
    main()