# -*- coding: utf-8 -*-
"""
下载 ChatTTS 权重（本地目录）
==============================
为什么需要单独脚本：

  ChatTTS 0.2.5 的 `chat.load(source="huggingface")` 走的是 GitHub release
  （RVC-Models-Downloader 的 rvcmd 下载器），国内网络基本连不上，
  会卡在 "downloading ..." 不动。

  0.2.5 对应的 HuggingFace 仓库是 `2Noise/ChatTTS`（不是老教程里的
  `hfl1995/chattts-pt`），里面已经是标准的 asset 目录结构，
  可以直接用 huggingface_hub 快照下载，然后用 source="local" 加载。

用法：
    venv_chattts\\Scripts\\python.exe scripts\\download_chattts_weights.py
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.info import MODEL_DIR, banner, footer, section

REPO = "2Noise/ChatTTS"
# ChatTTS 0.2.5 实际需要的文件（依据 ChatTTS/config/config.py 的 Path 定义
# 与 ChatTTS/res/sha256_map.json 的校验清单，不是老教程里的文件名）
#
#   vocos_ckpt_path   = asset/Vocos.safetensors
#   dvae_ckpt_path    = asset/DVAE.safetensors
#   decoder_ckpt_path = asset/Decoder.safetensors
#   embed_path        = asset/Embed.safetensors
#   gpt_ckpt_path     = asset/gpt/{config.json, model.safetensors}
#   tokenizer_path    = asset/tokenizer/{tokenizer.json, tokenizer_config.json,
#                                       special_tokens_map.json}
#
# ⚠️ 不要下 asset/*.pt 那套（GPT.pt / DVAE.pt / Decoder.pt …），
#    那是旧版格式，0.2.5 用 safetensors，且 GPT.pt 有 900MB 白下。
WANTED = [
    "asset/Vocos.safetensors",
    "asset/DVAE.safetensors",
    "asset/Decoder.safetensors",
    "asset/Embed.safetensors",
    "asset/gpt/config.json",
    "asset/gpt/model.safetensors",
    "asset/tokenizer/tokenizer.json",
    "asset/tokenizer/tokenizer_config.json",
    "asset/tokenizer/special_tokens_map.json",
]


def main():
    ap = argparse.ArgumentParser(description="下载 ChatTTS 权重")
    ap.add_argument("--dest", default=str(MODEL_DIR / "chattts-pt"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest)
    banner("下载 ChatTTS 权重")
    print(f"  仓库：{REPO}")
    print(f"  目标：{dest}")
    print(f"  镜像：{os.environ.get('HF_ENDPOINT')}")

    from huggingface_hub import snapshot_download

    section("开始下载")
    t0 = __import__("time").time()
    # 跳过已存在且大小合理的文件（GPT.pt 有 900MB，重下太慢）
    todo = []
    for rel in WANTED:
        p = dest / rel
        if p.exists() and p.stat().st_size > 1024:
            print(f"  [已存在] {rel}  {p.stat().st_size / 1e6:.0f} MB，跳过")
            continue
        todo.append(rel)

    if not todo:
        print("  所有文件均已存在。")
    else:
        try:
            path = snapshot_download(
                repo_id=REPO,
                local_dir=str(dest),
                allow_patterns=todo,
                max_workers=4,
            )
            print(f"  下载完成，用时 {__import__('time').time() - t0:.0f}s")
            print(f"  路径：{path}")
        except Exception as e:
            print(f"  [失败] {type(e).__name__}: {e}")
            print("  备用方案：手动到 https://hf-mirror.com/2Noise/ChatTTS 下载")
            print("            asset 目录下的全部文件，放到：")
            print(f"            {dest}\\asset")
            footer("下载失败")
            return

    section("文件清单")
    for p in sorted(dest.rglob("*")):
        if p.is_file() and ".cache" not in str(p):
            print(f"  {p.relative_to(dest)}  {p.stat().st_size / 1e6:.1f} MB")

    footer("ChatTTS 权重下载完成")


if __name__ == "__main__":
    main()
