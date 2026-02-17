import os
import numpy as np
import shutil
from .pose_get import RtmlibPoseGet
from typing import Optional, Mapping, Any
import inspect

from diffusers.utils import load_image
from utils.config import get_default_config
from .stage_context import StageContext
from .stage import Stage0Pose, Stage1BaseInpaint, Stage2CannyBackground, Stage3OpenPoseRefine

# pipeline loaders moved to separate module
from .pipeline_loaders import load_inpaint_pipe, load_controlnet_inpaint_pipe


class FourStageOutpainter:
    """
    Encapsulate stage0~stage3 pipelines and model loading.

    - Stage implementations are private methods: `_stage0/_stage1/_stage2/_stage3`.
    - Utilities such as `calculate_wh` and `make_canny_control` remain module-level.
    - Constructor accepts optional `pose_getter` and `pipe_*` instances and supports lazy loading.
    """

    def __init__(
        self,
        cfg=None,
        device: str = "cuda",
        pose_getter: Optional[object] = None,
        pipe_inpaint: Optional[object] = None,
        pipe_canny: Optional[object] = None,
        pipe_openpose: Optional[object] = None,
        lazy_load: bool = True,
        stage_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        self.cfg = cfg or get_default_config()
        self.device = device

        # model ids/paths from config
        self.animagine_model = self.cfg.animagine_xl
        self.base_model = self.cfg.diffusers_stable_diffusion_xl_inpainting_model
        self.ctl_model_canny = self.cfg.control_model_canny
        self.ctl_model_openpose = self.cfg.control_model_openpose
        self.vae_model = self.cfg.vae_model

        # optionally injected instances
        self.pose_getter = pose_getter
        self.pipe_inpaint = pipe_inpaint
        self.pipe_canny = pipe_canny
        self.pipe_openpose = pipe_openpose

        # lazy loading control
        self._models_loaded = False
        self._lazy = lazy_load

        if not self._lazy:
            self._lazy_load_models()

        stage_params = dict(stage_params or {})

        # stage pipeline (each stage returns a new StageContext)
        self.stages = [
            Stage0Pose(self, **self._stage_kwargs("stage0", Stage0Pose, padding=30, stage_params=stage_params)),
            Stage1BaseInpaint(
                self,
                **self._stage_kwargs(
                    "stage1",
                    Stage1BaseInpaint,
                    feather_radius=0,
                    guidance_scale=8.5,
                    num_inference_steps=30,
                    strength=1.0,
                    stage_params=stage_params,
                ),
            ),
            Stage2CannyBackground(
                self,
                **self._stage_kwargs(
                    "stage2",
                    Stage2CannyBackground,
                    controlnet_conditioning_scale=0.65,
                    guidance_scale=7.5,
                    num_inference_steps=35,
                    strength=0.90,
                    stage_params=stage_params,
                ),
            ),
            Stage3OpenPoseRefine(
                self,
                **self._stage_kwargs(
                    "stage3",
                    Stage3OpenPoseRefine,
                    controlnet_conditioning_scale=0.80,
                    guidance_scale=6.5,
                    num_inference_steps=40,
                    strength=0.56,
                    stage_params=stage_params,
                ),
            ),
        ]

        unknown_stage_keys = set(stage_params.keys()) - {"stage0", "stage1", "stage2", "stage3"}
        if unknown_stage_keys:
            raise ValueError(f"Unknown stage keys in stage_params: {sorted(unknown_stage_keys)}")

    @staticmethod
    def _stage_kwargs(
        stage_name: str,
        stage_cls: type,
        /,
        stage_params: Mapping[str, Mapping[str, Any]],
        **defaults: Any,
    ) -> dict[str, Any]:
        overrides = dict(stage_params.get(stage_name, {}) or {})

        # Validate override keys to avoid silent typos
        sig = inspect.signature(stage_cls.__init__)
        valid = {p.name for p in sig.parameters.values()} - {"self", "owner"}
        unknown = set(overrides.keys()) - valid
        if unknown:
            raise ValueError(
                f"Unknown params for {stage_name} ({stage_cls.__name__}): {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}"
            )

        merged = dict(defaults)
        merged.update(overrides)
        return merged

    def _lazy_load_models(self):
        if self._models_loaded:
            return

        print("🚀 正在准备模型路径和资源...")
        print("animagine_model:", self.animagine_model)
        print("base_model:", self.base_model)
        print("ctl_model_canny:", self.ctl_model_canny)
        print("ctl_model_openpose:", self.ctl_model_openpose)
        print("vae_model:", self.vae_model)

        print("\n🚀 正在加载模型...")

        if self.pose_getter is None:
            print("📌 加载 Pose 检测器...")
            self.pose_getter = RtmlibPoseGet()

        if self.pipe_inpaint is None:
            print("📌 加载 Base Inpaint Pipeline...")
            self.pipe_inpaint = load_inpaint_pipe(self.base_model, self.vae_model)

        if self.pipe_canny is None:
            print("📌 加载 Canny ControlNet...")
            self.pipe_canny = load_controlnet_inpaint_pipe(
                self.animagine_model, self.ctl_model_canny, self.vae_model
            )

        if self.pipe_openpose is None:
            print("📌 加载 OpenPose ControlNet...")
            self.pipe_openpose = load_controlnet_inpaint_pipe(
                self.animagine_model, self.ctl_model_openpose, self.vae_model
            )

        self._models_loaded = True
        print("✅ 模型加载完成\n")

    # -------------------------
    # Stage private methods
    # -------------------------
    # Orchestration
    # -------------------------
    def _execute_once(
        self,
        index: int,
        input_path: str,
        tmp_dir: str,
        save_path_template: str,
        debug: bool,
        prompt: str,
        negative_prompt: str,
    ):
        os.makedirs(tmp_dir, exist_ok=True)


        input_image = load_image(input_path).convert("RGB")
        context = StageContext(
            input_image=input_image,
            input_path=input_path,
            current_image=input_image,
            debug_dir=tmp_dir if debug else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

        # Stage 0
        print("\n=== Stage 0: 提取 Pose 和人物 Mask ===")
        context = self.stages[0].process(context)
        
        
        if debug:
            pose_path = os.path.join(tmp_dir, f"stage0_pose_{index}.png")
            context.pose_image.save(pose_path)
            print(f"Pose 保存: {pose_path}")
            mask = getattr(context, "person_mask", None)
            if mask is not None:
                mask_path = os.path.join(tmp_dir, f"stage0_mask_{index}.png")
                mask.save(mask_path)
                print(f"Mask 保存: {mask_path}")

        # Stage 1
        print("\n=== Stage 1: Base Inpaint 扩展背景 ===")
        context = self.stages[1].process(context)
        if debug:
            stage1_path = os.path.join(tmp_dir, f"stage1_inpaint_{index}.png")
            context.current_image.save(stage1_path)
            print(f"Stage1 保存: {stage1_path}")

            if context.canvas is not None:
                stage1_canvas_path = os.path.join(tmp_dir, f"stage1_canvas_{index}.png")
                context.canvas.save(stage1_canvas_path)
                print(f"Stage1 Canvas 保存: {stage1_canvas_path}")

        # Stage 2
        print("\n=== Stage 2: Canny 重构背景 ===")
        context = self.stages[2].process(context)
        if debug:
            stage2_path = os.path.join(tmp_dir, f"stage2_result_{index}.png")
            context.current_image.save(stage2_path)
            print(f"Stage2 保存: {stage2_path}")

        # Stage 3
        print("\n=== Stage 3: OpenPose 修正人物 ===")
        context = self.stages[3].process(context)

        save_path = save_path_template.format(index)
        context.current_image.save(save_path)
        print(f"\n✅ 完成第 {index} 次生成: {save_path}\n")

    def run(
        self,
        input_img_path,
        save_pic_path,
        prompt=None,
        negative_prompt=None,
        runs=5,
        debug=True,
    ):
        os.makedirs(save_pic_path, exist_ok=True)
        tmp_dir = os.path.join(save_pic_path, "tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        assert negative_prompt is None, "请提供 negative_prompt 参数以确保生成质量"
        negative_prompt = (
            "lowres, (bad anatomy, bad hands:1.2, bad legs:1.2), text, error, blurry"
            "missing fingers, extra digit, fewer digits, cropped, worst quality, "
            "low quality, normal quality, jpeg artifacts, signature, watermark, "
            "username"
        )

        save_path_template = os.path.join(save_pic_path, "res_{}.png")

        # Ensure models are available when first needed
        if not self._lazy:
            self._lazy_load_models()

        for i in range(1, runs + 1):
            print(f"\n{'=' * 60}")
            print(f"🎨 开始第 {i} 次四阶段扩图")
            print(f"{'=' * 60}")

            # load models on-demand right before executing
            if self._lazy and not self._models_loaded:
                self._lazy_load_models()

            self._execute_once(
                index=i,
                input_path=input_img_path,
                tmp_dir=tmp_dir,
                save_path_template=save_path_template,
                debug=debug,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )


def pic_expand(
    input_img_path,
    prompt,
    save_pic_path,
    debug: bool = False,
    stage_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
):
    """
    External interface: runs the four-stage outpaint and stores results under `save_pic_path`.

    stage_params: per-stage constructor overrides.

    Example:
        stage_params = {
            "stage0": {"padding": 20},
            "stage1": {"feather_radius": 2, "guidance_scale": 7.5, "strength": 0.95},
            "stage2": {"controlnet_conditioning_scale": 0.6, "strength": 0.9},
            "stage3": {"strength": 0.55},
        }
    """
    outpainter = FourStageOutpainter(stage_params=stage_params)
    return outpainter.run(
        input_img_path, 
        save_pic_path, 
        prompt=prompt, 
        runs=5, 
        debug=debug
    )


if __name__ == "__main__":
    shutil.rmtree("./tmp", ignore_errors=True)
    input_img_path = "/home/ping/Downloads/todo_pic/111965731_p0_master1200.jpg"
    prompt = (
        "little girl, sitting in a forest, sitting on a table, many plants, "
        "soft lighting, clean and minimalistic composition, emphasizing her "
        "cuteness and purity, highly detailed, ultra-realistic, 8k resolution, "
        "studio lighting, vibrant colors"
    )
    save_pic_path = "./run"
    pic_expand(
        input_img_path=input_img_path, prompt=prompt, save_pic_path=save_pic_path
    )
    if input("是否删除临时文件？(y/n): ") == "y":
        shutil.rmtree("./tmp", ignore_errors=True)
