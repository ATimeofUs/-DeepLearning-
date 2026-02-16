from huggingface_hub import HfApi, hf_hub_download
from .network_proxy import Network_proxy
import os
import subprocess

class Huggingface:
    def __init__(self, repo_id, open_clash_proxy=False):
        if open_clash_proxy:
            Network_proxy.open_agent("clash")

        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id)

        if open_clash_proxy:
            Network_proxy.close_agent()

        self.files = files
        self.repo_id = repo_id
        # __init__ must not return a value; files stored on the instance

    def show_url_list(self):
        print("=" * 30)
        print()
        print(f"Repo ID: {self.repo_id}")
        print(f"Len: {len(self.files)}")
        for file in self.files:
            print(file)
        print()
        print("=" * 30)

    def download_with_aria2(
        self,
        save_path: str,
        essential_files: list[str],
    ):
        # 创建目录
        os.makedirs(save_path, exist_ok=True)

        print("获取文件列表...")
        print(f"\n将下载 {len(essential_files)} 个文件到: {save_path}\n")

        # 逐个下载
        for i, file in enumerate(essential_files, 1):
            url = f"https://hf-mirror.com/{self.repo_id}/resolve/main/{file}"
            file_path = os.path.join(save_path, file)
            file_dir = os.path.dirname(file_path)

            # 创建子目录
            os.makedirs(file_dir, exist_ok=True)

            # 检查文件是否已存在
            if os.path.exists(file_path):
                print(f"[{i}/{len(essential_files)}] ✓ 已存在: {file}")
                continue

            print(f"[{i}/{len(essential_files)}] 下载中: {file}")

            # aria2c 命令
            cmd = [
                "aria2c",
                "-x",
                "16",  # 16线程
                "-s",
                "16",  # 分割成16段
                "-k",
                "1M",  # 最小分割大小1MB
                "--file-allocation=none",  # 不预分配
                "-d",
                file_dir,  # 保存目录
                "-o",
                os.path.basename(file),  # 文件名
                url,
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ]

            try:
                subprocess.run(cmd, check=True)
                print("    ✓ 完成\n")
            except subprocess.CalledProcessError as e:
                print(f"    ✗ 失败: {e}\n")
            except FileNotFoundError:
                print("错误: 未安装 aria2，请先安装: sudo pacman -S aria2")
                return

        print("\n" + "=" * 60)
        print("下载完成！")
        print("=" * 60)
        
        
    def download_with_official_tool(
        self,
        save_path: str,
        essential_files: list[str],
    ):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        print("\n" + "=" * 60)
        print(f"开始官方通道下载: {self.repo_id}")
        print(f"目标目录: {save_path}")
        print("=" * 60 + "\n")

        for i, file in enumerate(essential_files, 1):
            print(f"[{i}/{len(essential_files)}] 准备下载: {file}")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file,
                    local_dir=save_path,
                    local_dir_use_symlinks=False,  # 禁用符号链接，直接下载文件到目标文件夹
                    resume_download=True, # 支持断点续传        
                    endpoint="https://hf-mirror.com",
                )
                print(f"    ✓ 已完成: {downloaded_path}")
            except Exception as e:
                print(f"    ✗ 下载失败 {file}: {e}")
                print("    提示: 请检查网络连接或代理设置。")

        print("\n" + "=" * 60)
        print("所有任务处理完毕！")
        print("=" * 60)