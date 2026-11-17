"""下载 Huatuo-Lite 全量数据到 Huatuo-Lite/format_data.jsonl（服务器部署第一步）。

运行（仓库根目录）：
  python data/download_data.py            # 已存在则跳过
  python data/download_data.py --force    # 强制重新下载
国内默认走 hf-mirror 镜像（HF_ENDPOINT 已设时不覆盖）。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download

dst_dir = ROOT / "Huatuo-Lite"
dst = dst_dir / "format_data.jsonl"

if dst.exists() and "--force" not in sys.argv:
    print(f"已存在，跳过下载: {dst}（--force 强制重下）", flush=True)
    sys.exit(0)

path = hf_hub_download(
    repo_id="FreedomIntelligence/Huatuo26M-Lite",
    filename="format_data.jsonl",
    repo_type="dataset",
    local_dir=str(dst_dir),
)
print("下载完成:", path, flush=True)