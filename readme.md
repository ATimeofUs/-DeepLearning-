# 项目概览

这是一个面向图像扩展（outpaint）、人物修正与本地放大（upscale）的工具集。主要功能模块位于 `main_program`，各模块可独立运行或组合成流水线。模型文件可通过仓库内脚本下载，或在 `utils/config/config.py` 中修改模型路径后使用。

主要功能
- 基于 ControlNet + Stable Diffusion XL 的四阶段扩图流水线：Pose 提取 → 背景扩展 → Canny 重构 → OpenPose 人物修正
- 使用 rtmlib 提取人体关键点（openpose 风格）并生成用于 ControlNet 的条件图
- 本地 Real-ESRGAN（ncnn-vulkan）集成，用于离线图像放大与批处理
- 辅助脚本用于从 Hugging Face 下载模型文件，支持 aria2 加速或 huggingface 官方传输



