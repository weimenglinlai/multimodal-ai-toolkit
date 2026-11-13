# -*- coding: utf-8 -*-
"""
公用模块：个人信息输出、设备与路径管理。

说明要求截图中必须包含"个人信息输出"，所有脚本统一从这里取，
保证在每个脚本的输出里都能看到。
"""
from pathlib import Path
import os
import platform
import sys
import time

# ---- 个人信息（用于截图中的个人信息输出） ----
NAME = ""
STUDENT_ID = ""
CLASS_INFO = ""
LAB = "AI楼510 机器学习实践室"
PROJECT_NAME = "《AI》任务3 · 多模态大模型"

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
CLIP_CN_DIR = MODEL_DIR / "chinese-clip-vit-base-patch16"
CLIP_EN_DIR = MODEL_DIR / "clip-vit-base-patch32"
SD_DIR = MODEL_DIR / "tiny-sd"
WHISPER_HF_DIR = MODEL_DIR / "whisper-small"
PIC_DIR = ROOT / "clip_cn" / "pic"
SD_OUT_DIR = ROOT / "sd" / "outputs"
AUDIO_DIR = ROOT / "whisper" / "audio"
CHATTTS_OUT_DIR = ROOT / "chattts" / "outputs"
BIN_DIR = ROOT / "bin"
LOG_DIR = ROOT / "logs"

# ---- 缓存目录：一律落在项目内，不写 C 盘 ----
# 默认情况下 huggingface_hub 会把权重缓存到 ./data<用户>\.cache\huggingface，
# openai-whisper 会把 .pt 落到 ./data<用户>\.cache\whisper。
# 这里在导入任何模型库之前把环境变量改到项目内，保证本项目的所有中间产物
# 都待在项目目录下，不污染用户目录。
CACHE_DIR = MODEL_DIR / "cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
HF_HUB_DIR = HF_CACHE_DIR / "hub"
WHISPER_CACHE_DIR = CACHE_DIR / "whisper"
for _d in (HF_HUB_DIR, WHISPER_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_DIR)
# transformers / datasets / whisper 都尊重 XDG_CACHE_HOME
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)

for _d in (PIC_DIR, SD_OUT_DIR, AUDIO_DIR, CHATTTS_OUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 把项目自带的 ffmpeg 加进 PATH ----
# openai-whisper 的 load_audio() 无条件调用 ffmpeg 子进程解音频，
# 即使输入已经是 16kHz wav 也不例外（参考文档说"只处理 16kHz wav 可跳过 ffmpeg"实测不成立）。
# 这里直接用 imageio-ffmpeg 附带的二进制，不要求用户改系统环境变量。
_ffmpeg = BIN_DIR / "ffmpeg.exe"
if _ffmpeg.exists():
    _p = str(BIN_DIR)
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")


def banner(title: str) -> None:
    """打印统一抬头：含、、、实践名、时间。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "=" * 68
    print(line)
    print(f"  {title}")
    print(line)
    print(f"  姓    名：{NAME}")
    print(f"  学    号：{STUDENT_ID}")
    print(f"  院    系：{CLASS_INFO}")
    print(f"  实 验 室：{LAB}")
    print(f"  任务：{PROJECT_NAME}")
    print(f"  运行时间：{now}")
    print(f"  Python  ：{sys.version.split()[0]}  ({platform.python_implementation()})")
    print(line)


def footer(title: str = "任务结束") -> None:
    print("=" * 68)
    print(f"  {title}   —— {NAME} / {STUDENT_ID}")
    print("=" * 68)


def section(text: str) -> None:
    """小节标题"""
    print()
    print("-" * 68)
    print(f"  {text}")
    print("-" * 68)


def device_info() -> str:
    """返回 torch 设备描述字符串（CPU 或 GPU 型号）"""
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        return "CPU"
    except Exception:
        return "CPU"


def pick_device(prefer_gpu: bool = True):
    """返回 torch.device，prefer_gpu=True 时优先用 CUDA"""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
