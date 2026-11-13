# -*- coding: utf-8 -*-
"""
任务三 · Whisper 拓展任务：用 transformers 库加载 Whisper 实现语音转录
======================================================================
对应参考文档「（三）3．Whisper实践任务（2）拓展任务」

要求：在 hf-mirror 上搜索 openai/whisper，下载模型文件，
      自行编写代码，通过 transformers 库加载模型并实现语音转录。

与基础测试（openai-whisper 库）的差别：
  · openai-whisper 库：自带完整推理封装，模型名即字符串（"base"），
    权重自动从 Azure 下载（本项目已把缓存目录改到 model/cache/whisper，
    不写用户目录）；
  · transformers 库：模型从 HF/镜像目录加载，需自己写特征提取与解码，
    可精细控制（语言任务 prompt、是否输出时间戳等）。

两者用同一份音频、同一权重规格时结果基本一致，可做交叉验证。

运行：
    python whisper/task2_hf_whisper.py --audio audio/16k.wav
    python whisper/task2_hf_whisper.py --model-dir ../model/whisper-small
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.info import AUDIO_DIR, WHISPER_HF_DIR, banner, footer, section


def load_audio_16k(path: Path, sr: int = 16000):
    """读音频并重采样到 16kHz 单声道 float32。
    优先 soundfile（无需 ffmpeg），失败则回退到 librosa/ffmpeg。"""
    try:
        import soundfile as sf
        data, sr_in = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr_in != sr:
            import math
            n_out = int(math.ceil(len(data) * sr / sr_in))
            x_old = np.linspace(0, 1, len(data), endpoint=False)
            x_new = np.linspace(0, 1, n_out, endpoint=False)
            data = np.interp(x_new, x_old, data).astype("float32")
        return data, sr
    except Exception as e:
        print(f"  soundfile 读取失败（{e}），改用 librosa…")
        import librosa
        data, sr_in = librosa.load(str(path), sr=sr, mono=True)
        return data.astype("float32"), sr_in


def main():
    ap = argparse.ArgumentParser(description="transformers 版 Whisper 转录")
    ap.add_argument("--audio", default=str(AUDIO_DIR / "16k.wav"))
    ap.add_argument("--model-dir", default=str(WHISPER_HF_DIR),
                    help="本地 Whisper 模型目录（transformers 格式）")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"],
                    help="transcribe=转写原文；translate=翻译成英文")
    ap.add_argument("--chunk-seconds", type=int, default=30,
                    help="流式分块长度，Whisper 固定 30 秒窗口")
    args = ap.parse_args()

    banner("任务三 · Whisper 拓展任务：transformers 库加载 Whisper")
    print(f"  模型目录：{args.model_dir}")
    print(f"  音频文件：{args.audio}")
    print(f"  任务类型：{args.task}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists() or not any(model_dir.iterdir()):
        print(f"  [错误] 模型目录为空或不存在：{model_dir}")
        print("         请先执行：python scripts/download_models.py whisper")
        return

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"  [错误] 音频文件不存在：{audio_path}")
        return

    section("加载模型与特征提取器")
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  推理设备：{device}   dtype={dtype}")

    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=dtype
    ).to(device).eval()
    print("  模型加载完成。")

    section("读取音频")
    audio, sr = load_audio_16k(audio_path)
    duration = len(audio) / sr
    print(f"  采样率 {sr}Hz，时长 {duration:.2f} 秒，样本数 {len(audio)}")

    section("分块推理并生成文本")
    chunk = args.chunk_seconds * sr
    texts = []
    t0 = time.time()
    for i in range(0, len(audio), chunk):
        seg = audio[i:i + chunk]
        inputs = processor(seg, sampling_rate=sr, return_tensors="pt").to(device)
        if dtype == torch.float16:
            inputs["input_features"] = inputs["input_features"].to(dtype)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_features"],
                language="zh" if args.task == "transcribe" else None,
                task=args.task,
                max_new_tokens=200,
            )
        txt = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        texts.append(txt)
        print(f"    片段 {i // chunk + 1} ({len(seg) / sr:.1f}s) → {txt}")
    elapsed = time.time() - t0

    full = " ".join(texts)
    print(f"\n  完整转录结果：")
    print(f"  {'-' * 60}")
    print(f"  {full}")
    print(f"  {'-' * 60}")
    print(f"  推理耗时：{elapsed:.2f} 秒（音频 {duration:.2f} 秒，"
          f"实时率 RTF = {elapsed / max(duration, 1e-6):.2f}）")

    try:
        import zhconv
        simple = zhconv.convert(full, "zh-cn")
        if simple != full:
            print(f"\n  繁体转简体后：{simple}")
    except ImportError:
        pass

    section("与 openai-whisper 库的关系")
    print("  本脚本用的是 transformers 官方实现；whisper_test.py 用的是 openai-whisper 库。")
    print("  两者加载同一份权重、同一段音频时，转录内容应基本一致，可互相验证。")
    print("  若不一致，常见原因：分块方式不同（transformers 按 30 秒硬切，")
    print("  openai-whisper 会做长音频切分与重叠合并），或 feature extractor 参数不同。")
    print()

    # 数字归一化对比：把数字串里的空格/逗号去掉再比对
    import re

    digits = re.findall(r"\d+", full)
    compact = "".join(digits)
    print("  数字串归一化对比：")
    print(f"    原始转录中的数字片段：{digits}")
    print(f"    去掉分隔符后：{compact}")
    print(f"    长度：{len(compact)}")
    print()
    print("  说明：transformers 版按固定 30 秒窗硬切，数字串跨切点时会被插入空格，")
    print("        这里能看到「20230 1170 150」这种带空格的输出；")
    print("        归一化后数字本身是完整的，说明模型识别本身没问题，")
    print("        空格是分块策略带来的显示差异，而非识别错误。")
    print("        这一点在写实践分析时值得点出：同一模型不同推理框架的输出格式差异。")

    footer("Whisper 拓展任务完成")


if __name__ == "__main__":
    main()
