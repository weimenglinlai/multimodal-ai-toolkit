# -*- coding: utf-8 -*-
"""
任务三 · ChatTTS 子任务1：语音合成效果对比 —— 相同文本在不同平台的语音表现差异
================================================================================
对应参考文档「（三）5．ChatTTS实践任务（1）子任务1」

目的：探究同一文本在 ChatTTS 和 minimax 平台上合成语音的音质、语调、
      情感表达等方面的差异。

做法：
  1. 用 3 组测试文本（日常对话 / 新闻播报 / 情感表达），在本地用 ChatTTS
     合成，保存为 ChatTTS_*.wav；
  2. minimax 平台合成需要登录，本脚本不代劳，只负责：
       - 输出同样的 3 组文本，供你在 minimax 网页上逐条输入；
       - 检查你把 minimax 合成结果放到指定目录后，做一次自动对比统计
         （时长、采样率、响度 RMS 等客观指标）。

⚠️ ChatTTS 0.2.5 的 API（实测，与网上旧版示例不同）：
     from ChatTTS.core import Chat          # 类名是 Chat，不是 ChatTTS
     chat = Chat()
     chat.load(source="huggingface", device="cuda", use_flash_attn=False)
     spk = chat.sample_random_speaker()      # 音色 embedding
     wavs = chat.infer(text, params_infer_code=Chat.InferCodeParams(spk_emb=spk))
   旧版的 chat.infer(text, spk_spk=..., temperature=...) 在 0.2.5 已不存在。

运行：
    # 第一步：本地合成（需要 venv_chattts 环境）
    venv_chattts\\Scripts\\python.exe chattts\\chat_tts_synth.py

    # 第二步：把 minimax 合成结果放到 chattts/outputs/minimax/ 后
    venv_chattts\\Scripts\\python.exe chattts\\chat_tts_synth.py --compare
"""
import argparse
import math
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.info import CHATTTS_OUT_DIR, MODEL_DIR, banner, footer, section

# 参考文档指定的 3 组测试文本
TEST_TEXTS = [
    ("日常对话", "今天天气真好，我们一起去公园散步吧。"),
    ("新闻播报", "据最新报道，本市将于下周举办大型科技博览会，吸引众多企业参与。"),
    ("情感表达", "真没想到会在这里遇到你，我太开心了！"),
]

# minimax 下载结果应放这里
MINIMAX_DIR = CHATTTS_OUT_DIR / "minimax"
CHATTTS_DIR = CHATTTS_OUT_DIR / "chattts"
# 本地 ChatTTS 权重目录（由 scripts/download_chattts_weights.py 下载）
CHATTTS_MODEL_DIR = MODEL_DIR / "chattts-pt"


def _to_int16_bytes(wav) -> bytes:
    """把 ChatTTS 的波形转成 wav 文件要的 int16 PCM 字节

    ChatTTS.infer() 返回 float32 波形（-1~1）。
    直接 bytes() 会把浮点数的原始字节当样本写入 → 削波乱码，
    实测 peak=1.0 / RMS=0.54，Whisper 完全识别不出来。
    这里统一做 float→int16 量化（乘 32767 并裁剪）。
    """
    import numpy as np

    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    arr = np.asarray(wav).reshape(-1)

    if arr.dtype == np.int16:
        return arr.tobytes()
    if arr.dtype == np.int32:
        return (arr >> 16).astype(np.int16).tobytes()

    # 浮点：量化到 int16
    arr = np.clip(arr.astype(np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def _get_sample_rate(chat, default: int = 24000) -> int:
    """从 ChatTTS 实例里取采样率（0.2.5 里路径很深，多级兜底）"""
    # 1) 直接属性（旧版本）
    sr = getattr(chat, "samplerate", None)
    if isinstance(sr, int) and sr > 0:
        return sr
    # 2) config.vocos.feature_extractor.init_args.sample_rate
    try:
        sr = chat.config.vocos.feature_extractor.init_args.sample_rate
        if isinstance(sr, int) and sr > 0:
            return sr
    except AttributeError:
        pass
    # 3) 递归找任意 sample_rate 属性
    def _walk(obj, depth=0):
        if depth > 5 or obj is None:
            return None
        v = getattr(obj, "sample_rate", None)
        if isinstance(v, int) and v > 0:
            return v
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if isinstance(child, (int, float, str, bool)):
                continue
            if hasattr(child, "__dict__") or hasattr(child, "init_args"):
                r = _walk(child, depth + 1)
                if r:
                    return r
        return None

    return _walk(getattr(chat, "config", None)) or default


def synth_local(texts, use_gpu: bool, spk: str, local_dir: str = None):
    """用 ChatTTS 本地合成，返回 {标签: 路径}

    local_dir 不为空时用 source="local" 从本地目录加载权重
    （ChatTTS 0.2.5 的 source="huggingface" 走 GitHub release，国内连不上）。
    """
    from ChatTTS.core import Chat

    device = "cuda" if use_gpu else "cpu"
    print(f"  加载 ChatTTS（device={device}）…")
    t0 = time.time()
    chat = Chat()
    if local_dir:
        ok = chat.load(source="local", custom_path=local_dir, device=device,
                       use_flash_attn=False)
    else:
        ok = chat.load(source="huggingface", device=device, use_flash_attn=False)
    if not ok:
        print("  [错误] ChatTTS 权重加载失败")
        print("         请先执行：")
        print("           venv_chattts\\Scripts\\python.exe scripts\\download_chattts_weights.py")
        return {}
    print(f"  加载完成，用时 {time.time() - t0:.1f} 秒。")

    # 音色 embedding
    if not spk:
        spk = chat.sample_random_speaker()
        print(f"  随机音色 spk = {spk[:32]}…")

    CHATTTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, text in texts:
        print(f"\n  合成 [{tag}]：{text}")
        t1 = time.time()
        params = Chat.InferCodeParams(spk_emb=spk, temperature=0.3)
        wavs = chat.infer(text, params_infer_code=params)
        elapsed = time.time() - t1

        if not wavs:
            print("    [失败] 未返回音频")
            continue
        wav = wavs[0]
        # ChatTTS 0.2.5 的采样率藏在 config 的多层里：
        #   chat.config.vocos.feature_extractor.init_args.sample_rate = 24000
        # Chat 对象上没有 samplerate 属性，config 上也没有直接的 sample_rate。
        sr = _get_sample_rate(chat)
        # ⚠️ 关键：infer() 返回的是 float32 波形（取值 -1~1），
        #    不能直接 bytes() 当 int16 写进 wav —— 那样写出来是削波乱码，
        #    实测 peak=1.0、RMS=0.54，Whisper 识别结果全空。
        #    必须先做 float → int16 量化。
        wav = _to_int16_bytes(wav)
        path = CHATTTS_DIR / f"ChatTTS_{tag}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(wav)
        dur = len(wav) / sr / 2
        print(f"    耗时 {elapsed:.2f} 秒，时长 {dur:.2f} 秒，采样率 {sr} Hz")
        print(f"    已保存：{path.name}")
        out[tag] = path
    return out


def wav_stats(path: Path):
    """读取 wav 的客观指标：时长、采样率、峰值、RMS(dBFS)"""
    try:
        import numpy as np
    except ImportError:
        return None
    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    dur = len(data) / sr
    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
    rms_db = 20 * math.log10(rms) if rms > 1e-9 else -99.0
    return {"dur": dur, "sr": sr, "peak": peak, "rms_db": rms_db}


def compare():
    """对比 ChatTTS 与 minimax 的客观指标"""
    section("客观指标对比（ChatTTS vs minimax）")
    print("  说明：以下为可量化的客观指标，主观听感请配合参考文档要求的截图自行判断。")
    print()

    rows = []
    for tag, text in TEST_TEXTS:
        c_path = CHATTTS_DIR / f"ChatTTS_{tag}.wav"
        m_path = MINIMAX_DIR / f"minimax_{tag}.wav"
        c = wav_stats(c_path) if c_path.exists() else None
        m = wav_stats(m_path) if m_path.exists() else None
        rows.append((tag, c, m))

    print(f"  {'类别':<12}{'ChatTTS 时长':>14}{'minimax 时长':>14}"
          f"{'CTTS RMS(dB)':>15}{'MMIN RMS(dB)':>15}")
    print("  " + "-" * 70)
    for tag, c, m in rows:
        cd = f"{c['dur']:.2f}s" if c else "—"
        md = f"{m['dur']:.2f}s" if m else "待下载"
        cr = f"{c['rms_db']:.1f}" if c else "—"
        mr = f"{m['rms_db']:.1f}" if m else "—"
        print(f"  {tag:<14}{cd:>14}{md:>14}{cr:>15}{mr:>15}")

    missing = [t for t, c, m in rows if m is None]
    if missing:
        section("待办")
        print(f"  以下类别的 minimax 音频还没放进 {MINIMAX_DIR}：")
        for t in missing:
            print(f"    · {t}  → 需要命名为 minimax_{t}.wav")
        print("  完成后重新运行本脚本 --compare 即可看到完整对比表。")
    else:
        section("观察要点（供报告分析）")
        print("  1. 时长差异：同文本不同平台语速不同，长度差越大说明断句/语速处理差异越明显。")
        print("  2. RMS 响度：minimax 通常做了响度归一化，ChatTTS 未做时 RMS 会明显偏低。")
        print("  3. 结合主观听感补全音质、语调、情感表达、流畅度四个维度的分析。")

    footer("对比统计完成")


def main():
    ap = argparse.ArgumentParser(description="ChatTTS 语音合成与 minimax 对比")
    ap.add_argument("--compare", action="store_true", help="只做对比统计，不合成")
    ap.add_argument("--texts", nargs="+", default=None,
                    help="自定义文本（默认用参考文档的 3 组）")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 合成")
    args = ap.parse_args()

    banner("任务三 · ChatTTS 子任务1：语音合成效果对比")

    texts = TEST_TEXTS
    if args.texts:
        texts = [(f"自定义{i + 1}", t) for i, t in enumerate(args.texts)]

    if args.compare:
        compare()
        return

    section("测试文本")
    for tag, text in texts:
        print(f"  [{tag}] {text}")

    section("加载 ChatTTS")
    try:
        import torch
        use_gpu = torch.cuda.is_available() and not args.cpu
    except ImportError:
        use_gpu = False
    print(f"  推理设备：{'GPU' if use_gpu else 'CPU'}")

    try:
        import ChatTTS  # noqa: F401
    except ImportError:
        print("  [错误] 未安装 ChatTTS。")
        print("         ChatTTS 依赖 transformers 4.x，与本项目主环境（transformers 5.x）冲突，")
        print("         请使用项目内独立环境：")
        print("           venv_chattts\\Scripts\\python.exe chattts\\chat_tts_synth.py")
        print("         独立环境的安装脚本见 scripts\\setup_chattts_env.bat")
        return

    section("开始合成")
    local_dir = str(CHATTTS_MODEL_DIR) if CHATTTS_MODEL_DIR.exists() else None
    if local_dir:
        print(f"  使用本地权重：{local_dir}")
    else:
        print("  [提示] 未找到本地权重目录，将尝试 source=\"huggingface\"")
        print("         （该路径走 GitHub release，国内可能卡住，")
        print("          建议先跑 scripts\\download_chattts_weights.py）")
    synth_local(texts, use_gpu, None, local_dir)

    section("minimax 平台合成（需手动操作）")
    print("  minimax 在线地址：https://www.minimaxi.com/audio/text-to-speech")
    print("  操作步骤：")
    for tag, text in texts:
        print(f"    1) 打开上述页面，选择女声音色")
        print(f"    2) 输入文本：{text}")
        print(f"    3) 点击合成并下载，保存为 minimax_{tag}.wav")
    print(f"  下载后把 3 个文件放入：{MINIMAX_DIR}")
    print("  然后运行：python chattts/chat_tts_synth.py --compare")

    footer("ChatTTS 合成完成")


if __name__ == "__main__":
    main()
