项目名称

基于SDXL模型的多阶段人物扩图生成系统（SDXL + ControlNet Pipeline）

简介

本项目实现了一个模块化的多阶段人物扩图（outpainting）流水线，基于 Stable Diffusion XL 的 inpainting 能力并结合 ControlNet 条件（如深度、Canny 边缘与 OpenPose 姿态），实现人物姿态保持、画布扩展、背景重建与细节修复。目标是为输入人物图像生成自然且与原人物保持一致性的扩展画面，适用于图片编辑、图像增强与素材生成场景。

主要特性

- 三阶段推理流水线：
	- Stage1: Base Inpaint（创建扩展画布、羽化掩码、基础修复）
	- Stage2: Depth + Canny（使用深度图与边缘作为 ControlNet 条件补全背景）
	- Stage3: Tile + Pose（基于分块与姿态控制细化人物与背景细节）
- 模块化设计：各阶段实现统一的 `Stage` 接口，通过 `StageContext` 传递上下文数据，支持替换不同的 ControlNet 模型或 pose/深度提取方法。
- 依赖注入：通过 `StageDependencies` / `_ThreeStageOutpaintDeps` 提供管线加载与 pose 获取接口，便于测试和扩展。
- 延迟加载（lazy load）：支持模型按需加载以降低显存占用。
- 调试支持：`StageContext` 包含 `debug_dir`，各阶段可输出中间控制图与中间结果以便排查。

技术栈

- 语言/框架：Python、PyTorch
- 生成模型：Diffusers、Stable Diffusion XL (SDXL)
- 条件控制：ControlNet（Depth / Canny / OpenPose / Tile 等）
- 姿态/控制图：OpenPose 风格提取、项目内的 `control_image_get` 工具
- 推理优化：ONNX Runtime（可选）
- 工具：OpenCV / PIL 等图像处理库

代码结构（概览）

- `ThreeStageOutpaint`：主控制类，负责配置、设备、模型/管线加载、并按顺序执行 Stage 列表。
- `Stage` 抽象类：定义 `process(context: StageContext)` 接口，具体阶段实现继承该类。
- `StageContext`：承载 `input_image`、`current_image`、`canvas`、`expand_mask`、`pose_image`、`prompt`、`debug_dir` 等运行时数据。
- `utils/my_diffusers/control_image_get.py`：提供深度、Canny、tile 与 pose 控制图的生成函数和绘制工具。
- `utils/my_diffusers/pipeline_loaders.py`：封装 diffusers 管线加载（inpaint / multi-controlnet 等）。
- `pic_expand`：上层调用入口函数，用于接收输入路径、prompt、negative prompt、输出路径与调试选项并运行流水线。

快速开始（示例）

1. 安装（示例依赖）

```bash
pip install torch diffusers transformers accelerate safetensors opencv-python onnxruntime
```

2. 简单调用示例（项目内接口）

```python
from utils.my_diffusers.pic_expand import pic_expand

pic_expand(
		input_img_path='input.jpg',
		prompt='A photorealistic portrait extended with natural background',
		negative_prompt='',
		save_pic_path='out.png',
		debug=True,
		stage_params=None,
)
```

3. 可选：配置与模型路径在 `utils/config/config.py` 中管理（请根据环境修改）。

设计与实现要点

- 保持人物姿态：Stage3 使用 pose 条件与阈值控制避免人物形变。
- 多条件融合：Stage2 将深度与边缘条件结合以补全结构性背景信息，Stage3 用 tile/pose 细化纹理与局部细节。
- 可替换的 ControlNet：通过依赖注入接口可更换不同的 ControlNet 权重或添加新的控制类型。

调试与扩展建议

- 在 `StageContext.debug_dir` 下输出中间控制图（depth/canny/pose）以便观察各阶段效果。
- 若显存受限，启用 lazy load 并在不同阶段释放不再需要的模型或显存缓存。
- 若要提高速度，可尝试导出关键模型为 ONNX 并使用 `onnxruntime` 推理。

参考文件

- 类图：`utils/my_diffusers/detail/类图.mmd`
- 说明（本文件）：`utils/my_diffusers/detail/readme.md`

下一步

如果需要，我可以：
- 将此 README 英文化并生成 `README.md` 到项目根目录；
- 为项目生成 `requirements.txt` 或 `pyproject.toml`；
- 添加一个最小的运行脚本和示例图片以便快速验证。

版权与许可证

此文档仅为项目说明示例；请在仓库中添加适当的 LICENSE 文件以明确版权和使用许可。