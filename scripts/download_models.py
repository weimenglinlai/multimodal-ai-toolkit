# -*- coding: utf-8 -*-
"""
任务3 · 模型下载脚本
通过 hf-mirror.com 国内镜像下载全部实践所需模型。

用法：
    python scripts/download_models.py            # 下载全部
    python scripts/download_models.py clip sd    # 只下载指定部分
    python scripts/download_models.py --list     # 查看清单
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# ---- 走 hf-mirror 镜像 ----
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- 需要下载的模型清单 ----
# allow_patterns 控制只取必要文件，避免把仓库里的大权重重复拉一遍
MODELS = {
    "clip": {
        "repo_id": "OFA-Sys/chinese-clip-vit-base-patch16",
        "local_dir": MODEL_DIR / "chinese-clip-vit-base-patch16",
        "allow_patterns": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "*.md",
        ],
        "modelscope": "AI-ModelScope/chinese-clip-vit-base-patch16",
        "desc": "中文CLIP base（任务一主力模型）",
    },
    "clip_en": {
        "repo_id": "openai/clip-vit-base-patch32",
        "local_dir": MODEL_DIR / "clip-vit-base-patch32",
        "allow_patterns": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        # HF 通道不稳时的备用源（国内直连）
        "modelscope": "openai-mirror/clip-vit-base-patch32",
        "desc": "OpenAI 原版 CLIP（拓展任务1 对比用）",
    },
    "sd": {
        "repo_id": "segmind/tiny-sd",
        "local_dir": MODEL_DIR / "tiny-sd",
        "allow_patterns": [
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "unet/*",
            "vae/*",
        ],
        "desc": "segmind/tiny-sd 轻量版Stable Diffusion（任务二）",
    },
    "whisper": {
        "repo_id": "openai/whisper-small",
        "local_dir": MODEL_DIR / "whisper-small",
        "allow_patterns": [
            "config.json",
            "preprocessor_config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        "desc": "HuggingFace 版 Whisper（拓展任务用）",
    },
}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def download_from_modelscope(repo_id: str, local: Path, files: list) -> bool:
    """
    备用通道：用 ModelScope 快照下载（国内直连，通常比 HF 稳）。

    repo_id 形如 "owner/name"，文件从 repo 根目录取。
    """
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
    except ImportError:
        print("      [跳过] 未安装 modelscope")
        return False

    try:
        ms_download(
            repo_id=repo_id,
            local_dir=str(local),
            allow_patterns=files,
        )
    except Exception as e:
        print(f"      [失败] ModelScope 通道异常: {type(e).__name__}: {e}")
        return False
    return True


def download_via_curl(repo_id: str, local: Path, files: list) -> bool:
    """
    兜底通道：用 curl 直连 hf-mirror 的 resolve 端点。
    实测 xet 协议下载 >500MB 文件会反复超时中断，curl 走普通 HTTP 稳定得多。
    """
    script = Path(__file__).resolve().parent / "curl_download.py"
    if not script.exists():
        print("      [跳过] 未找到 curl_download.py")
        return False

    cmd = [sys.executable, str(script), repo_id] + files + ["--dest", str(local)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("      [失败] curl 通道未能完成下载")
        return False
    except Exception as e:
        print(f"      [失败] curl 通道异常：{e}")
        return False


def download_one(key: str, force: bool = False) -> bool:
    cfg = MODELS[key]
    local = Path(cfg["local_dir"])
    print("=" * 70)
    print(f"[{key}] {cfg['repo_id']}")
    print(f"      说明: {cfg['desc']}")
    print(f"      目标: {local}")

    if local.exists() and dir_size(local) > 50 * 1024 * 1024 and not force:
        print(f"      已存在（{human(dir_size(local))}），跳过。加 --force 可强制重下。")
        return True

    t0 = time.time()
    ok = False
    try:
        snapshot_download(
            repo_id=cfg["repo_id"],
            local_dir=str(local),
            allow_patterns=cfg.get("allow_patterns"),
            max_workers=4,
        )
        ok = True
    except Exception as e:
        print(f"      [HF 通道失败] {type(e).__name__}: {e}")
        print("      改走 curl 直连通道 …")
        ok = download_via_curl(cfg["repo_id"], local, cfg.get("allow_patterns") or ["*"])

    size = dir_size(local)
    print(f"      用时 {time.time() - t0:.0f}s，大小 {human(size)}")
    if not ok or size < 1024 * 1024:
        print("      [警告] 下载未完成或内容过小，请检查。")
        return False
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        for k, v in MODELS.items():
            print(f"{k:10s} {v['repo_id']:45s} {v['desc']}")
        return

    keys = args if args else list(MODELS.keys())
    force = "--force" in sys.argv

    ok, fail = [], []
    for k in keys:
        if k not in MODELS:
            print(f"未知模型键: {k}")
            continue
        (ok if download_one(k, force) else fail).append(k)

    print("=" * 70)
    print(f"成功 {len(ok)} 个: {ok}")
    if fail:
        print(f"失败 {len(fail)} 个: {fail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
