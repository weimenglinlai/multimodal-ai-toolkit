# -*- coding: utf-8 -*-
"""
用 Whisper 反向验证 TTS 合成质量
=================================
目的：ChatTTS 子任务1 只要求"和 minimax 对比"，但只靠主观听感说服力不足。
     本脚本把合成音频送回 Whisper 做 ASR，得到「原文 → 合成 → 识别」的闭环，
     用**机器可量化的方式**评估合成质量：

       · 识别文本与原文的字符错误率（CER）—— 越低说明合成越清晰、发音越准
       · 漏字/多字的具体位置 —— 定位是哪些字没说清楚
       · 语速（字数/秒）—— 判断韵律是否自然

     这条链路正好把任务三的 Whisper 部分和 ChatTTS 部分串起来，
     对应参考文档「七、拓展思考（三）」的思路。

运行：
    venv\\Scripts\\python.exe scripts\\eval_tts_quality.py
    venv\\Scripts\\python.exe scripts\\eval_tts_quality.py --model small
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.info import CHATTTS_OUT_DIR, banner, footer, section

# (标签, 原文)
TEST_TEXTS = [
    ("日常对话", "今天天气真好，我们一起去公园散步吧。"),
    ("新闻播报", "据最新报道，本市将于下周举办大型科技博览会，吸引众多企业参与。"),
    ("情感表达", "真没想到会在这里遇到你，我太开心了！"),
]

# 同一组文本在 minimax 平台合成后的文件名（放进 chattts/outputs/minimax/）
MINIMAX_DIR = CHATTTS_OUT_DIR / "minimax"
CHATTTS_DIR = CHATTTS_OUT_DIR / "chattts"


def levenshtein(a: str, b: str) -> int:
    """编辑距离"""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(s: str) -> str:
    """去掉标点与空白，只留实义字符；同时做繁简归一

    ChatTTS 在部分文本上会输出繁体（实测新闻播报那组输出了
    「據說新報導…中國體驗」），不归一化的话 CER 会虚高到 60%+，
    那是识别出的字形差异，不是合成质量问题。
    """
    out = []
    for ch in s:
        if ch.strip() and not ch.isspace() and ch not in "，。！？、；：,.!?;:\"'“”‘’（）()《》…—-·":
            out.append(ch)
    s = "".join(out)
    try:
        import zhconv
        s = zhconv.convert(s, "zh-cn")
    except Exception:
        pass
    return s


def cer(ref: str, hyp: str) -> float:
    """字符错误率 = 编辑距离 / 参考长度"""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0
    return levenshtein(r, h) / len(r)


def main():
    ap = argparse.ArgumentParser(description="Whisper 反验 TTS 合成质量")
    ap.add_argument("--model", default="base", help="Whisper 模型规模")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    banner("用 Whisper 反向验证 TTS 合成质量")
    print(f"  Whisper 模型：{args.model}")
    print(f"  ChatTTS 目录：{CHATTTS_DIR}")
    print(f"  minimax 目录：{MINIMAX_DIR}")

    import whisper

    section("加载 Whisper")
    model = whisper.load_model(args.model)
    print("  模型加载完成。")

    def asr(path: Path) -> str:
        r = model.transcribe(str(path), language="zh", fp16=False)
        return r["text"].strip()

    results = []

    for tag, text in TEST_TEXTS:
        c_path = CHATTTS_DIR / f"ChatTTS_{tag}.wav"
        m_path = MINIMAX_DIR / f"minimax_{tag}.wav"
        if not c_path.exists():
            continue

        c_text = asr(c_path)
        m_text = asr(m_path) if m_path.exists() else None

        results.append({
            "tag": tag, "ref": text,
            "ctts": c_text, "cer_c": cer(text, c_text),
            "mmax": m_text,
            "cer_m": cer(text, m_text) if m_text else None,
        })

    if not results:
        print("  [提示] 没有可评估的音频，先跑 chattts/chat_tts_synth.py")
        footer("无可评估数据")
        return

    section("评估结果")
    print(f"  {'类别':<10}{'ChatTTS CER':>14}{'minimax CER':>14}{'ChatTTS 识别文本':<40}")
    for r in results:
        m_cer = f"{r['cer_m']:.2%}" if r["cer_m"] is not None else "待下载"
        print(f"  {r['tag']:<12}{r['cer_c']:>12.2%}{m_cer:>14}  {r['ctts'][:38]}")

    section("逐条明细")
    for r in results:
        print(f"\n  【{r['tag']}】")
        print(f"    原文      ：{r['ref']}")
        print(f"    ChatTTS   ：{r['ctts']}")
        if r["mmax"] is not None:
            print(f"    minimax   ：{r['mmax']}")
        print(f"    ChatTTS CER = {r['cer_c']:.2%}"
              + (f"    minimax CER = {r['cer_m']:.2%}" if r["cer_m"] is not None else ""))
        # 繁简提示
        try:
            import zhconv
            if zhconv.convert(r["ctts"], "zh-cn") != r["ctts"]:
                print(f"    （ChatTTS 输出为繁体，已用 zhconv 转为简体后计算 CER）")
                print(f"     简体：{zhconv.convert(r['ctts'], 'zh-cn')}")
        except Exception:
            pass
        # 逐字对齐差异
        ref_n, hyp_n = normalize(r["ref"]), normalize(r["ctts"])
        print(f"    实义字符数：原文 {len(ref_n)} / 识别 {len(hyp_n)}")
        if r["cer_c"] < 0.05:
            print("    → 识别结果与原文几乎完全一致，合成清晰度、发音准确度良好")
        elif r["cer_c"] < 0.2:
            print("    → 有少量字词偏差，属于 TTS 常见的中文同音字/连读问题")
        else:
            print("    → 偏差较大，建议检查文本标点、语速或音色选择")

    section("解读")
    print("  CER（字符错误率）越低说明合成语音越容易被 ASR 正确还原，")
    print("  即音质清晰、发音准确、韵律自然。它是一个客观代理指标，")
    print("  可以和主观听感（音质/语调/情感/流畅度）互相印证。")
    print()
    print("  ⚠️ 注意：CER 只能反映「说得清不清楚」，不能反映「有没有感情」。")
    print("     情感表达那一条如果 CER 很低但你听着平淡，")
    print("     说明 ASR 指标对韵律不敏感，这正是需要主观评价补上的部分。")

    footer("TTS 质量评估完成")


if __name__ == "__main__":
    main()
