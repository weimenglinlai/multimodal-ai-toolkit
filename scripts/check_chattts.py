# -*- coding: utf-8 -*-
"""
ChatTTS 环境专项检查
====================
主环境的 transformers 是 5.x，ChatTTS 跑不了；
本脚本用 venv_chattts 的解释器运行，检查 ChatTTS 是否可用。

运行：
    venv_chattts\\Scripts\\python.exe scripts\\check_chattts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.info import MODEL_DIR, banner, footer, section


def main():
    banner("ChatTTS 环境检查")

    section("包版本")
    ok = True
    for name in ["torch", "transformers", "ChatTTS", "numba", "numpy",
                 "pybase16384", "vector_quantize_pytorch", "vocos"]:
        try:
            m = __import__(name)
            ver = getattr(m, "__version__", "")
            print(f"  √ {name:<26} {ver}")
            if name == "transformers":
                major = int(ver.split(".")[0])
                if major >= 5:
                    print("    [警告] transformers 是 5.x，ChatTTS 需要 4.x")
                    ok = False
        except Exception as e:
            print(f"  × {name:<26} 缺失 ({type(e).__name__})")
            ok = False

    section("CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  √ CUDA 可用：{torch.cuda.get_device_name(0)}")
        else:
            print("  - 无 CUDA，ChatTTS 将在 CPU 上跑（很慢）")
    except Exception:
        pass

    section("权重")
    # ChatTTS 0.2.5 用本地目录加载（source="local"），权重在 model/chattts-pt。
    # 注意仓库是 2Noise/ChatTTS，早期教程里的 hfl1995/chattts-pt 已经不适用；
    # 需要的也是 safetensors 那套，不是 asset/*.pt。
    weight_dir = MODEL_DIR / "chattts-pt"
    need = [
        "asset/gpt/model.safetensors",
        "asset/Embed.safetensors",
        "asset/DVAE.safetensors",
        "asset/Decoder.safetensors",
        "asset/Vocos.safetensors",
    ]
    missing = [r for r in need if not (weight_dir / r).exists()]
    if not missing:
        size = sum(f.stat().st_size for f in weight_dir.rglob("*") if f.is_file())
        print(f"  √ ChatTTS 权重已就位（{size / 1024 / 1024:.0f} MB）")
        print(f"    {weight_dir}")
    else:
        print(f"  × 权重不完整，缺 {len(missing)} 个文件：")
        for r in missing:
            print(f"      {r}")
        print("    修复：venv_chattts\\Scripts\\python.exe scripts\\download_chattts_weights.py")

    section("结论")
    if ok:
        print("  ChatTTS 环境就绪。运行：")
        print("    venv_chattts\\Scripts\\python.exe chattts\\chat_tts_synth.py")
    else:
        print("  环境不完整，请执行：scripts\\setup_chattts_env.bat")

    footer("检查完成")


if __name__ == "__main__":
    main()
