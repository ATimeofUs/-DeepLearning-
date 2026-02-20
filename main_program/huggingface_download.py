import os
from utils.web.huggingface import Huggingface

def op0(repo_id):
    hf = Huggingface(repo_id=repo_id, open_clash_proxy=True)
    for f in hf.files:
        print(f)


def op1(repo_id, save_path, required_files=None):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    hf = Huggingface(
        repo_id=repo_id,
        open_clash_proxy=True
    )
    
    hf.download_with_official_tool(
        save_path=save_path, 
        essential_files=required_files,
    )


if __name__ == "__main__":
    op = input("请输入命令")

    save_path = "/home/ping/model/meta-llama-Llama-3.1-8B"
    # repo_id = "black-forest-labs/FLUX.1-Fill-dev"
    repo_id = "meta-llama/Llama-3.1-8B"

    required_files = [
        "README.md",
        "USE_POLICY.md",
        "config.json",
        "generation_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "original/consolidated.00.pth",
        "original/params.json",
        "original/tokenizer.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]


    if op == "0":
        op0(repo_id)
    elif op == "1":
        op1(repo_id, save_path, required_files)
    else:
        print("无效命令")

    input("Press Enter to continue...")
    
"""


meta-llama/Llama-3.1-8B 

xinsir/controlnet-depth-sdxl-1.0

YarvixPA/FLUX.1-Fill-dev-GGUF

flux1-fill-dev-Q4_K_S.gguf

"""