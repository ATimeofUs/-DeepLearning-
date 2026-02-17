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

    save_path = "/run/media/ping/TRAIN_DATA/model/pic/blip-image-captioning-large"
    repo_id = "Salesforce/blip-image-captioning-large"

    required_files = [
        "README.md",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
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