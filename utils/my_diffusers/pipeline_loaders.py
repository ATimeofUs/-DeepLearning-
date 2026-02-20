import torch
from diffusers import (
    ControlNetModel, # ctl 模型
    StableDiffusionXLControlNetInpaintPipeline, # 带 ctl 的 inpaint SDXL pipeline
    AutoencoderKL, # Stable Diffusion 的 VAE 模型
    StableDiffusionXLInpaintPipeline, # 不带 ctl 的 inpaint SDXL pipeline
)


def load_inpaint_pipe(base_model, vae_model, lora_model=None):
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    if lora_model is not None:
        pipe.load_lora_weights(
            lora_model, 
            adapter_name="lora",
            torch_dtype=torch.float16
        )
        pipe.fuse_lora(lora_scale=0.5, adapter_names=["lora"])
    
    pipe.enable_sequential_cpu_offload()
    return pipe


def load_controlnet_inpaint_pipe(base_model, control_model, vae_model, lora_model=None):
    controlnet = ControlNetModel.from_pretrained(
        control_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    if lora_model is not None:
        pipe.load_lora_weights(
            lora_model, 
            adapter_name="lora",
            torch_dtype=torch.float16
        )
        pipe.set_adapters(["lora"], adapter_weights=[0.4])

    pipe.enable_sequential_cpu_offload()
    return pipe


def load_multi_controlnet_inpaint_pipe(
    base_model,
    control_models: list[str],
    vae_model,
    lora_model=None,
):
    controlnets = [
        ControlNetModel.from_pretrained(
            model,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        for model in control_models
    ]

    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=controlnets,   
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    if lora_model is not None:
        pipe.load_lora_weights(
            lora_model,
            adapter_name="lora",
            torch_dtype=torch.float16,
        )
        pipe.set_adapters(["lora"], adapter_weights=[0.4])

    pipe.enable_sequential_cpu_offload()
    return pipe