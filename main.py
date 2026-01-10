from ultralytics.models.yolo import YOLO

pre = YOLO('yolov8n.pt')            # 预训练权重（标准结构）
model = YOLO('GProject/my.yaml')    # 自定义结构（包含 TransformerLayer）

if model.model is not None:
    raise ValueError("模型已初始化，无法加载预训练权重。请在创建模型时指定权重文件。")
model.model.load_state_dict(pre.model.state_dict(), strict=False)

# 然后继续训练微调
model.train(data='path/to/your_dataset.yaml', epochs=30, imgsz=640)