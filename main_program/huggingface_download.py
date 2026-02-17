import os
from utils.web.huggingface import Huggingface

def op0(repo_id):
    hf = Huggingface(repo_id=repo_id, open_clash_proxy=True)
    for f in hf.files:
        print(f)


def op1(repo_id, save_path, required_files=None):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    hf = Huggingface(
        repo_id=repo_id,
        open_clash_proxy=True
    )
    
    hf.download_with_aria2(
        save_path=save_path, 
        essential_files=required_files,
    )


if __name__ == "__main__":
    op = input("请输入命令")

    save_path = "/run/media/ping/TRAIN_DATA/model/pic/animagine-xl-4.0"
    repo_id = "cagliostrolab/animagine-xl-4.0"

    required_files = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.safetensors",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer_2/tokenizer_config.json",
        "scheduler/scheduler_config.json"
    ]

    if op == "0":
        op0(repo_id)
    elif op == "1":
        op1(repo_id, save_path, required_files)
    else:
        print("无效命令")

    input("Press Enter to continue...")
    
"""

"""