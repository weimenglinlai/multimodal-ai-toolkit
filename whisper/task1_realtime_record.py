# -*- coding: utf-8 -*-
"""
任务三 · Whisper 子任务1：麦克风录音 + 实时语音识别
==================================================
对应参考文档「（三）3．Whisper实践任务（1）子任务1」

控制方式：空格开始/停止录音，ESC 退出程序。
要求：录音时说出自己的，输出中要能看到识别出的。

依赖：
    pip install pyaudio keyboard soundfile numpy
  · pyaudio 用于麦克风采集（Windows 需 PortAudio，一般 pip 预编译包可用）
  · keyboard 用于全局热键监听（空格/ESC）
  · 若 pyaudio 装不上，可用 sounddevice 采集，见本文件内 fallback 分支。

运行：
    python whisper/task1_realtime_record.py
    python whisper/task1_realtime_record.py --model base --device cpu
"""
import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.info import (AUDIO_DIR, NAME, STUDENT_ID, WHISPER_CACHE_DIR, banner,
                         footer, section)

SAMPLE_RATE = 16000
CHANNELS = 1


def check_keywords(text: str):
    """检查与是否被正确识别

    ASR 对连续数字串经常丢位或误识（如  → 20230117010），
    所以不只做"完全相等"判断，还用最长公共子串长度量化接近程度。
    """
    import re

    name_ok = NAME in text.replace(" ", "")
    print(f"  关键词「{NAME}」：{'✓ 已识别' if name_ok else '✗ 未识别到'}")

    digits = re.findall(r"\d+", text)
    sid_compact = STUDENT_ID
    exact = any(sid_compact in d for d in digits)

    best_lcs = 0
    for d in digits:
        # 最长公共子串 DP：dp[i][j] 存二维，别用一维滚动（会丢状态）
        n, m = len(d), len(sid_compact)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if d[i - 1] == sid_compact[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > best_lcs:
                        best_lcs = dp[i][j]
                else:
                    dp[i][j] = 0

    if exact:
        print(f"  关键词「{STUDENT_ID}」：✓ 已完整识别")
    else:
        print(f"  关键词「{STUDENT_ID}」：✗ 未完整识别")
        if digits:
            print(f"    ASR 输出的数字串：{digits}")
            print(f"    与正确的最长公共子串长度：{best_lcs}/{len(sid_compact)}"
                  f"（{best_lcs / len(sid_compact) * 100:.0f}%）")
            near = best_lcs >= len(sid_compact) - 2
            print(f"    → 说明识别结果{'仅差个别数字' if near else '偏差较大'}，"
                  f"属于 Whisper 对长数字串的常见识别误差，可换 --model small 改善")
        else:
            print("    → 识别结果里没有数字，可能被读成了中文数字或被跳过")


def record_to_wav(out_path: Path, seconds: float, samplerate=SAMPLE_RATE):
    """用 pyaudio 录音并保存为 16kHz 单声道 wav"""
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS,
                      rate=samplerate, input=True, frames_per_buffer=1024)
    frames = []
    print(f"  录音中…（最长 {seconds:.0f} 秒，说完后按空格停止）")
    t0 = time.time()
    while time.time() - t0 < seconds:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
        elapsed = time.time() - t0
        peak = int(max(abs(int.from_bytes(b, "little", signed=True))
                        for b in frames[-1][:64])) if frames[-1] else 0
        print(f"\r  已录 {elapsed:5.1f}s / {seconds:.0f}s   当前音量峰值 {peak:>6d}", end="")
    stream.stop_stream()
    stream.close()
    pa.terminate()

    raw = b"".join(frames)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(raw)
    print()
    return out_path


def record_to_wav_sounddevice(out_path: Path, seconds: float, samplerate=SAMPLE_RATE):
    """pyaudio 不可用时的替代方案：用 sounddevice 采集"""
    import sounddevice as sd

    print(f"  录音中…（最长 {seconds:.0f} 秒，说完后按空格停止）")
    t0 = time.time()
    chunks = []
    while time.time() - t0 < seconds:
        chunk = sd.rec(int(0.1 * samplerate), samplerate=samplerate, channels=CHANNELS,
                       dtype="int16")
        sd.wait()
        chunks.append(chunk)
        print(f"\r  已录 {time.time() - t0:5.1f}s / {seconds:.0f}s", end="")
    print()
    data = np.concatenate(chunks, axis=0)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(data.tobytes())
    return out_path


def main():
    ap = argparse.ArgumentParser(description="麦克风录音 + Whisper 实时识别")
    ap.add_argument("--model", default="base")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="单次最长录音时长")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    banner("任务三 · Whisper 子任务1：麦克风录音 + 实时语音识别")
    print(f"  模型规格：{args.model}")
    print(f"  录音参数：{SAMPLE_RATE}Hz / {CHANNELS}声道 / 16bit PCM")
    print(f"  保存目录：{AUDIO_DIR}")
    print(f"  请在录音时清晰说出：「{NAME}」和「{STUDENT_ID}」")

    section("加载模型")
    import whisper

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    model = whisper.load_model(args.model, device=device,
                               download_root=str(WHISPER_CACHE_DIR))
    print(f"  模型加载完成（device={device}）。")

    section("等待操作")
    print("  [空格] 开始录音        [空格] 停止录音并识别        [ESC] 退出")

    try:
        import keyboard
    except ImportError:
        print("  [错误] 未安装 keyboard，pip install keyboard")
        return

    out_path = AUDIO_DIR / "record.wav"
    recording = False
    round_no = 0

    def on_press(key):
        nonlocal recording, round_no
        if key == "space":
            if not recording:
                recording = True
                print("\n  >>> 空格按下，开始录音…")
                try:
                    record_to_wav(out_path, args.max_seconds)
                except Exception as e:
                    print(f"  [错误] pyaudio 录音失败：{e}")
                    print("        改用 sounddevice 方案…")
                    try:
                        record_to_wav_sounddevice(out_path, args.max_seconds)
                    except Exception as e2:
                        print(f"  [错误] sounddevice 也失败：{e2}")
                        recording = False
                        return
                recording = False
                round_no += 1
                print(f"\n  >>> 录音结束（第 {round_no} 次），开始识别…")
                do_transcribe(model, out_path)
        elif key == "esc":
            print("\n  >>> ESC 按下，退出程序。")
            keyboard.release_all()
            sys.exit(0)

    def do_transcribe(model, path):
        t0 = time.time()
        result = model.transcribe(str(path), language="zh", fp16=False)
        elapsed = time.time() - t0
        text = result["text"].strip()
        print(f"\n  识别耗时 {elapsed:.2f} 秒")
        print(f"  识别结果：{text}")
        try:
            import zhconv
            text = zhconv.convert(text, "zh-cn")
        except Exception:
            pass
        print()
        check_keywords(text)
        print("  （若未识别到，请提高音量/靠近麦克风后重试，或把模型换成 small 提升准确率）")

    keyboard.on_press(on_press)
    print("  监听已启动，等待按键…（ESC 退出）")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        keyboard.release_all()

    footer("实时录音识别结束")


if __name__ == "__main__":
    main()
