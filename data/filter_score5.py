"""过滤 score=5 并原位替换 Huatuo-Lite/format_data.jsonl（服务器部署第二步，低内存机型用）。

运行（仓库根目录）：
  python data/filter_score5.py              # 原位替换；首次自动留 format_data.jsonl.full 备份
  python data/filter_score5.py 源jsonl 目标jsonl   # 只写目标、不动源（本机测试用）
替换后 build_index.py 与后端读同一标准文件名，无需改代码/环境变量；重跑 build_index 即得 score5 索引。
恢复全量：把 format_data.jsonl.full 复制回 format_data.jsonl 再重跑 build_index。
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Huatuo-Lite" / "format_data.jsonl"
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src

kept = []
n = 0
with open(src, encoding="utf-8") as f:
    for line in f:
        n += 1
        if json.loads(line).get("score") == 5:
            kept.append(line)
        if n % 50000 == 0:
            print(f"scan {n} kept {len(kept)}", flush=True)

if dst == src:
    bak = src.with_suffix(src.suffix + ".full")
    if not bak.exists():
        shutil.copyfile(src, bak)
        print("已备份全量 ->", bak, flush=True)
    tmp = src.with_suffix(src.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        out.writelines(kept)
    os.replace(tmp, src)
else:
    with open(dst, "w", encoding="utf-8") as out:
        out.writelines(kept)

print(f"done total={n} score5={len(kept)} -> {dst}", flush=True)