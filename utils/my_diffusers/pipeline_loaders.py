import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    AutoencoderKL,
    StableDiffusionXLInpaintPipeline,
)


def load_inpaint_pipe(base_model, vae_model):
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
    pipe.enable_sequential_cpu_offload()
    return pipe


def load_controlnet_inpaint_pipe(base_model, control_model, vae_model):
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
    pipe.enable_sequential_cpu_offload()
    return pipe
