# -*- coding: utf-8 -*-
"""
任务3 环境一键检查
=====================
检查所有依赖、模型、素材是否就位，并给出缺失项的修复命令。

运行：
    python scripts/check_env.py
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.info import (AUDIO_DIR, CACHE_DIR, CLIP_CN_DIR, CLIP_EN_DIR,
                         HF_CACHE_DIR, HF_HUB_DIR, MODEL_DIR, PIC_DIR, ROOT,
                         SD_DIR, WHISPER_CACHE_DIR, WHISPER_HF_DIR,
                         banner, footer, section)

# (包名, 是否必需, 最低版本要求说明)
PACKAGES = [
    ("torch", True, "需要支持 CUDA（本项目实测 2.6.0+cu124）"),
    ("transformers", True, "5.x 可用（ChineseCLIP / CLIP / Whisper 均支持）"),
    ("diffusers", True, "0.42.1 实测可用"),
    ("accelerate", True, "diffusers 加载用"),
    ("PIL", True, "Pillow，处理图片"),
    ("numpy", True, ""),
    ("soundfile", True, "读写 wav"),
    ("whisper", True, "openai-whisper，Whisper 转录"),
    ("zhconv", True, "繁简转换"),
    ("keyboard", True, "实时录音热键监听"),
    ("sounddevice", False, "pyaudio 不可用时的录音替代"),
    ("pyaudio", False, "麦克风录音（Windows 需 PortAudio）"),
    ("safetensors", True, ""),
]

# (标签, 目录, 必需文件, 权重文件最小字节数)
# 最小字节数用来识别"下了一半"的残file（hf-mirror xet 超时会留下截断文件）
MODELS = [
    ("中文CLIP (任务一)", CLIP_CN_DIR,
     ["config.json", "pytorch_model.bin", "vocab.txt", "preprocessor_config.json"],
     {"pytorch_model.bin": 750 * 1024 * 1024}),
    ("原版CLIP (拓展1)", CLIP_EN_DIR,
     ["config.json", "pytorch_model.bin", "vocab.json", "merges.txt"],
     {"pytorch_model.bin": 600 * 1024 * 1024}),
    ("tiny-sd (任务二)", SD_DIR,
     ["model_index.json", "unet/config.json", "vae/config.json",
      "text_encoder/config.json", "tokenizer/vocab.json"],
     {"unet/diffusion_pytorch_model.bin": 640 * 1024 * 1024,
      "vae/diffusion_pytorch_model.bin": 165 * 1024 * 1024,
      "text_encoder/pytorch_model.bin": 240 * 1024 * 1024}),
    ("Whisper-HF (拓展)", WHISPER_HF_DIR,
     ["config.json", "model.safetensors", "tokenizer.json"],
     {"model.safetensors": 960 * 1024 * 1024}),
]

# 远端文件大小（字节），用于精确比对是否下载完整
# 这些是从 hf-mirror 实际 HEAD 拿到的 Content-Length
REMOTE_SIZES = {
    ("chinese-clip-vit-base-patch16", "pytorch_model.bin"): 753177983,
    ("clip-vit-base-patch32", "pytorch_model.bin"): 605247071,
    ("tiny-sd", "unet/diffusion_pytorch_model.bin"): 646923637,
    ("tiny-sd", "vae/diffusion_pytorch_model.bin"): 167407857,
    ("tiny-sd", "text_encoder/pytorch_model.bin"): 246187869,
    ("whisper-small", "model.safetensors"): 966995080,
}


def check_packages():
    section("依赖包检查")
    missing = []
    for name, required, note in PACKAGES:
        mod = name
        if name == "PIL":
            mod = "PIL"
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            status = f"OK   {ver}"
            ok = True
        except Exception as e:
            status = f"缺失 ({type(e).__name__})"
            ok = False
            if required:
                missing.append(name)
        flag = "√" if ok else ("×" if required else "-")
        print(f"  {flag} {name:<16} {status:<24} {note}")
    return missing


def check_models():
    section("模型文件检查")
    missing = []
    for label, folder, files, min_sizes in MODELS:
        if not folder.exists():
            print(f"  × {label:<22} 目录不存在：{folder}")
            missing.append(label)
            continue
        lack = [f for f in files if not (folder / f).exists()]
        # 权重文件大小校验（识别下载中断留下的残file）
        # 优先用远端 Content-Length 精确比对；没有记录时退回宽松阈值
        for rel, min_size in min_sizes.items():
            p = folder / rel
            if not p.exists():
                lack.append(rel)
                continue
            actual = p.stat().st_size
            expected = REMOTE_SIZES.get((folder.name, rel))
            if expected is not None:
                # 允许 1 字节误差
                if actual < expected - 1:
                    print(f"  × {label:<22} {rel} 体积异常："
                          f"{actual / 1024 / 1024:.0f}MiB < 远端 {expected / 1024 / 1024:.0f}MiB"
                          f"（下载中断，需重下）")
                    lack.append(rel)
            elif actual < min_size:
                print(f"  × {label:<22} {rel} 体积异常："
                      f"{actual / 1024 / 1024:.0f}MiB < 期望 {min_size / 1024 / 1024:.0f}MiB")
                lack.append(rel)
        if lack:
            print(f"  × {label:<22} 缺文件：{', '.join(sorted(set(lack)))}")
            missing.append(label)
        else:
            size = sum(f.stat().st_size for f in folder.rglob("*")
                       if f.is_file() and ".cache" not in str(f))
            print(f"  √ {label:<22} 完整（{size / 1024 / 1024:.0f} MB）")
    return missing


def check_assets():
    section("素材检查")
    imgs = [p for p in PIC_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if PIC_DIR.exists() else []
    print(f"  {'√' if imgs else '×'} CLIP 测试图片        {len(imgs)} 张（目录 {PIC_DIR}）")
    wavs = [p for p in AUDIO_DIR.glob("*.wav")] if AUDIO_DIR.exists() else []
    print(f"  {'√' if wavs else '×'} Whisper 测试音频     {len(wavs)} 个（目录 {AUDIO_DIR}）")

    # 缓存路径核对：必须全部落在项目内，不能写用户目录
    section("缓存路径检查")
    ok_cache = True
    for label, var, expect in [
        ("HF_HOME", "HF_HOME", HF_CACHE_DIR),
        ("HF_HUB_CACHE", "HF_HUB_CACHE", HF_HUB_DIR),
        ("XDG_CACHE_HOME", "XDG_CACHE_HOME", CACHE_DIR),
    ]:
        actual = os.environ.get(var, "")
        good = Path(actual) == expect if actual else False
        ok_cache = ok_cache and good
        print(f"  {'√' if good else '×'} {label:<16} {actual or '(未设置)'}")
    print(f"  · Whisper 权重目录     {WHISPER_CACHE_DIR}")
    if not ok_cache:
        print("    修复：直接通过项目脚本运行即可，common/info.py 会自动设置这些变量")
    return (not imgs), (not wavs)


def check_gpu():
    section("GPU 检查")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  √ CUDA 可用：{torch.cuda.get_device_name(0)}")
            print(f"    torch {torch.__version__}  CUDA {torch.version.cuda}")
            print(f"    显存 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
            return True
        print("  - 未检测到 CUDA，程序将回退到 CPU（速度慢很多，但可运行）")
        return False
    except Exception as e:
        print(f"  × torch 不可用：{e}")
        return False


def main():
    banner("任务3 · 环境检查")
    print(f"  项目根目录：{ROOT}")

    check_gpu()
    pkg_missing = check_packages()
    model_missing = check_models()
    no_img, no_wav = check_assets()

    section("结论")
    if not pkg_missing and not model_missing and not no_img:
        print("  环境就绪，可以直接运行各实践脚本。")
    else:
        if pkg_missing:
            print(f"  缺依赖包：{pkg_missing}")
            print("    修复：venv\\Scripts\\python.exe -m pip install " + " ".join(pkg_missing))
        if model_missing:
            print(f"  缺模型：{model_missing}")
            print("    修复：python scripts\\download_models.py")
        if no_img:
            print("  缺测试图片")
            print("    修复：python scripts\\download_test_images.py")
        if no_wav:
            print("  缺测试音频（可选，Whisper 实践需要）")
            print("    修复：python scripts\\make_test_audio.py")
            print("    或自己录一段 16kHz 单声道 wav 放到 whisper\\audio\\")

    section("脚本清单")
    for rel in [
        "clip_cn/test_clip.py            任务一 基础测试",
        "clip_cn/task1_1N_match.py       任务一 子任务1：1-N 文图匹配",
        "clip_cn/task2_MN_match.py       任务一 子任务2：M-N 文图匹配",
        "clip_cn/task3_other_clip_compare.py  任务一 拓展1：原版CLIP对比",
        "sd/test_tiny_gen_one.py         任务二 基础测试",
        "sd/task1_prompt_engineering.py  任务二 子任务1：提示词工程",
        "sd/task2_inference_steps.py     任务二 子任务2：推理步数权衡",
        "sd/task3_clip_eval.py           任务二 拓展1：CLIP相似度评估",
        "whisper/whisper_test.py         任务三 基础测试：语音转录",
        "whisper/task1_realtime_record.py 任务三 子任务1：麦克风实时识别",
        "whisper/task2_hf_whisper.py     任务三 拓展：transformers版Whisper",
        "chattts/chat_tts_synth.py       任务三 ChatTTS合成与minimax对比",
    ]:
        p = ROOT / rel.split()[0]
        mark = "√" if p.exists() else "×"
        print(f"  {mark} {rel}")

    footer("环境检查完成")


if __name__ == "__main__":
    main()
