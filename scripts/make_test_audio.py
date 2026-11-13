# -*- coding: utf-8 -*-
"""
生成 Whisper 测试用音频
=======================
目的：任务三需要一个 16kHz 单声道 wav 作为 Whisper 的输入。
      参考文档的 16k.wav 属于教师素材，本脚本用本地 TTS 合成一段
      含的中文语音，生成规格与要求完全一致的音频。

做法：优先用 Windows 自带的 SAPI（pyttsx3 或 win32com），
      无中文语音时退到 ChatTTS；最后统一用 soundfile 重采样为
      16kHz 单声道 16bit PCM wav。

运行：
    python scripts/make_test_audio.py
    python scripts/make_test_audio.py --text "我叫，"
"""
import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from common.info import AUDIO_DIR, NAME, STUDENT_ID, banner, footer, section

DEFAULT_TEXT = f"我叫{NAME}，{STUDENT_ID}。今天天气很好，我们一起去做实践吧。"


def synth_sapi(text: str, out_wav: Path, rate: int = 22050) -> bool:
    """用 Windows SAPI 合成（win32com），优先选中文语音"""
    try:
        import win32com.client
    except ImportError:
        print("  SAPI 方案不可用：未安装 pywin32")
        return False

    try:
        speak = win32com.client.Dispatch("SAPI.SpVoice")
    except Exception as e:
        print(f"  SAPI 初始化失败：{e}")
        return False

    # 找一个中文语音
    voices = []
    for i in range(speak.GetVoices().Count):
        v = speak.GetVoices().Item(i)
        try:
            desc = v.GetDescription()
        except Exception:
            desc = ""
        voices.append((v, desc))
        if any(k in desc for k in ("Chinese", "中文", "Yun", "Huihui", "Xiaoxiao", "Kangkang")):
            speak.Voice = v
            print(f"  选用语音：{desc}")
            break
    else:
        print(f"  未找到中文语音，使用默认语音：{speak.Voice.GetDescription()}")

    # SAPI 输出到 wav 文件
    # 注意：SpFileStream.Open 接受的是普通文件路径，不能用 file:// URI
    # （实测 file URI 会抛 com_error -2147024809）
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Open(str(out_wav.resolve()), 3, False)  # 3 = SSFM_CREATE_FORWRITE
    speak.AudioOutputStream = stream
    speak.Rate = 0
    speak.Speak(text)
    stream.Close()
    return out_wav.exists() and out_wav.stat().st_size > 1000


def synth_pyttsx3(text: str, out_wav: Path) -> bool:
    """pyttsx3 方案"""
    try:
        import pyttsx3
    except ImportError:
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.save_to_file(text, str(out_wav))
        engine.runAndWait()
        return out_wav.exists() and out_wav.stat().st_size > 1000
    except Exception as e:
        print(f"  pyttsx3 失败：{e}")
        return False


def resample_to_16k(src: Path, dst: Path) -> bool:
    """任意格式 → 16kHz 单声道 16bit wav"""
    try:
        import soundfile as sf
        data, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            n_out = int(round(len(data) * 16000 / sr))
            x_old = np.linspace(0, 1, len(data), endpoint=False)
            x_new = np.linspace(0, 1, n_out, endpoint=False)
            data = np.interp(x_new, x_old, data).astype("float32")
        pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())
        return True
    except Exception as e:
        print(f"  重采样失败：{e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="生成 Whisper 测试音频")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", default="16k.wav")
    args = ap.parse_args()

    banner("生成 Whisper 测试音频")
    print(f"  文本内容：{args.text}")
    print(f"  输出规格：16000 Hz / 单声道 / 16 bit PCM wav")

    final = AUDIO_DIR / args.out
    raw = AUDIO_DIR / "_raw_sapi.wav"

    section("合成语音")
    ok = False
    if synth_sapi(args.text, raw):
        print("  SAPI 合成成功。")
        ok = True
    else:
        if synth_pyttsx3(args.text, raw):
            print("  pyttsx3 合成成功。")
            ok = True

    if not ok:
        print("  [错误] 本机 TTS 方案均不可用。")
        print("         替代做法：")
        print("           1) 用手机录音或 ChatTTS 合成一段中文语音；")
        print("           2) 用 Audacity / ffmpeg 转成 16kHz 单声道 wav；")
        print("           3) 放到 whisper/audio/16k.wav 后继续。")
        print("         ffmpeg 命令：ffmpeg -i input.m4a -ar 16000 -ac 1 -acodec pcm_s16le 16k.wav")
        return

    section("转换为 16kHz 单声道 wav")
    if not resample_to_16k(raw, final):
        print("  [错误] 转换失败。")
        return
    raw.unlink(missing_ok=True)

    with wave.open(str(final), "rb") as wf:
        dur = wf.getnframes() / wf.getframerate()
        print(f"  已保存：{final}")
        print(f"  采样率 {wf.getframerate()} Hz，声道 {wf.getnchannels()}，"
              f"位深 {wf.getsampwidth() * 8} bit，时长 {dur:.2f} 秒")

    section("下一步")
    print(f"  转录测试：python whisper/whisper_test.py --audio {final}")
    print(f"  HF 版转录：python whisper/task2_hf_whisper.py --audio {final}")

    footer("测试音频生成完成")


if __name__ == "__main__":
    main()
