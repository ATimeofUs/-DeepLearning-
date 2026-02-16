# 项目概览

这是一个面向图像扩展（outpaint）、人物修正与本地放大（upscale）的工具集。主要功能模块位于 `main_program`，各模块可独立运行或组合成流水线。模型文件可通过仓库内脚本下载，或在 `utils/config/config.py` 中修改模型路径后使用。

主要功能
- 基于 ControlNet + Stable Diffusion XL 的四阶段扩图流水线：Pose 提取 → 背景扩展 → Canny 重构 → OpenPose 人物修正
- 使用 rtmlib 提取人体关键点（openpose 风格）并生成用于 ControlNet 的条件图
- 本地 Real-ESRGAN（ncnn-vulkan）集成，用于离线图像放大与批处理
- 辅助脚本用于从 Hugging Face 下载模型文件，支持 aria2 加速或 huggingface 官方传输

主要文件与简要说明
- `main_program/HD_restoration.py`：
	- 实现了一个基于 PyQt6 的本地 GUI 工具，用于调用 `realesrgan-ncnn-vulkan` 或本地放大模型执行图像放大。
	- 功能包括：选择输入文件/文件夹、选择输出路径、设置放大倍数/模型/分块(tile)/GPU 编号/启用 TTA/输出格式等选项；实时日志面板；以及 `UpscaleWorker` 后台线程用于运行命令并通过信号上报进度与完成状态。
- `main_program/pic_expand.py`：
	- 实现了完整的四阶段扩图/重绘流水线，包括画布计算、扩展区域 mask 生成、人物 Pose 与 mask 提取（基于 `RtmlibPoseGet`）、Stage1 背景 inpaint、Stage2 基于 Canny 的 ControlNet 背景重构、Stage3 基于 OpenPose 的 ControlNet 人物修正。
	- 提供辅助函数（`calculate_wh`、`create_canvas_and_mask`、`make_canny_control`、pipeline 加载等）与主流程函数 `run_four_stage_outpaint`，支持调试输出与配置读取。
- `main_program/hf_download.py`：
	- 简单的命令行脚本，用于列出 Hugging Face 仓库文件或使用 aria2 下载模型文件。脚本示例展示了如何设置 `HF_ENDPOINT` 镜像与启用 HF_TRANSFER，再调用仓库封装的 `Huggingface` 下载工具。
- `utils/config/config.py`：
	- 项目配置入口，管理模型路径、运行时选项与默认参数，请根据本机环境调整模型目录和缓存路径。

快速开始
1. 进入仓库根目录：
```bash
cd /path/to/repo
```
2. 安装依赖（根据需要）：
```bash
pip install -r requirements.txt
```
3. 配置模型路径：编辑 `utils/config/config.py` 中的模型目录设置，或使用仓库内的 `main_program/hf_download.py` 下载模型。
4. 运行示例：
- 本地放大 GUI：运行 `python main_program/HD_restoration.py`
- 四阶段扩图流程示例：运行 `python main_program/pic_expand.py`（根据脚本内的 `main()` 使用说明）
- 下载模型：运行 `python main_program/hf_download.py` 并按照提示选择操作

注意事项
- 运行扩图与生成模型通常需要配置合适的 GPU、CUDA 与 `torch` 环境。
- 下载大型模型建议使用 aria2 或镜像源以节省时间与失败重试成本。

贡献与许可证
- 欢迎提交 Issue 或 PR。请在公开发布前补充 LICENSE 文件以明确授权。

如需我把 README 翻译为英文、添加使用示例或 CI 配置，我可以继续完善。



