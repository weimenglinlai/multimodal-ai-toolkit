# -*- coding: utf-8 -*-
"""
任务三 · Whisper 语音转录 —— 基础测试程序
=========================================
对应参考文档「（三）2．使用 Whisper 进行语音转录（2）代码编辑和运行」

功能：加载 Whisper 模型，对一段中文音频做转录，输出转录文本与分段信息，
      并用 zhconv 把可能出现的繁体中文转成简体。

音频要求：16kHz 单声道 WAV（参考文档说明：只处理 16kHz wav 可跳过 ffmpeg）。
      如果音频是其它格式，需先安装 ffmpeg 并让它在 PATH 中。

运行：
    python whisper/whisper_test.py --audio audio/16k.wav
    python whisper/whisper_test.py --audio audio/16k.wav --model base
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.info import AUDIO_DIR, WHISPER_CACHE_DIR, banner, footer, section

DEFAULT_AUDIO = AUDIO_DIR / "16k.wav"


def main():
    ap = argparse.ArgumentParser(description="Whisper 语音转录基础测试")
    ap.add_argument("--audio", default=str(DEFAULT_AUDIO), help="待转录音频路径")
    ap.add_argument("--model", default="base", help="模型规模：tiny/base/small/medium/large-v3")
    ap.add_argument("--language", default="zh", help="语言代码，zh=中文")
    ap.add_argument("--no-zhconv", action="store_true", help="不做繁简转换")
    args = ap.parse_args()

    banner("任务三 · Whisper 语音转录（基础测试）")
    print(f"  音频文件：{args.audio}")
    print(f"  模型规格：{args.model}")
    print(f"  语言    ：{args.language}")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"  [错误] 音频文件不存在：{audio_path}")
        print("         请把 16kHz 单声道 wav 放到 whisper/audio/ 目录后再运行。")
        return

    section("加载 Whisper 模型")
    t0 = time.time()
    try:
        import whisper
    except ImportError:
        print("  [错误] 未安装 openai-whisper，请执行：pip install openai-whisper")
        return

    model = whisper.load_model(args.model, download_root=str(WHISPER_CACHE_DIR))
    print(f"  模型加载完成，用时 {time.time() - t0:.1f} 秒。")
    print(f"  权重缓存路径：{WHISPER_CACHE_DIR}")

    section("开始转录")
    t1 = time.time()
    result = model.transcribe(str(audio_path), language=args.language, fp16=False)
    elapsed = time.time() - t1

    print(f"\n  result['text'] 完整转录文本：")
    print(f"  {'-' * 60}")
    text = result["text"].strip()
    print(f"  {text}")
    print(f"  {'-' * 60}")
    print(f"\n  转录耗时：{elapsed:.2f} 秒")

    section("分段信息 result['segments']")
    for i, seg in enumerate(result.get("segments", []), 1):
        print(f"  [{i:>2}] {seg['start']:>7.2f}s ~ {seg['end']:>7.2f}s  {seg['text'].strip()}")

    # 繁简转换
    if not args.no_zhconv:
        section("繁体转简体（zhconv）")
        try:
            import zhconv

            simple = zhconv.convert(text, "zh-cn")
            print(f"  转换前：{text}")
            print(f"  转换后：{simple}")
            if simple != text:
                print("  → 检测到繁体内容，已转换。")
            else:
                print("  → 本段已是简体中文，无需转换。")
        except ImportError:
            print("  未安装 zhconv，跳过。pip install zhconv")

    section("识别检查")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task1_realtime_record import check_keywords

    check_keywords(text)

    footer("Whisper 基础测试完成")


if __name__ == "__main__":
    main()
