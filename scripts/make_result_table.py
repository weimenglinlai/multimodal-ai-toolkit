# -*- coding: utf-8 -*-
"""
生成实践数据汇总表（Markdown + CSV）
====================================
把各子任务的实测数据汇总成一张表，直接贴进报告第五节「实践结果与分析」。

数据来源：各脚本运行时写到 logs/ 下的日志，或重新计算。
本脚本不重跑模型，只做「读日志 → 抽数据 → 出表」。

运行：
    python scripts/make_result_table.py
    python scripts/make_result_table.py --out 结果汇总.md
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.info import LOG_DIR, NAME, STUDENT_ID, banner, footer, section


def parse_task2_steps(log_path: Path):
    """从子任务2日志里抽 步数/耗时 表"""
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    # 日志里的行形如： "        10          1.01           101.0           1.00x"
    # 注意行尾没有空格，用 \s*$ 会误伤，改用行首缩进 + 4 个字段
    for m in re.finditer(r"^\s+(\d{1,4})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s*$",
                         text, re.M):
        rows.append({
            "steps": int(m.group(1)),
            "total": float(m.group(2)),
            "per_step_ms": float(m.group(3)),
            "ratio": m.group(4),
        })
    return rows


def parse_task3_eval(log_path: Path):
    """从拓展1日志里抽 步数/相似度/耗时"""
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for m in re.finditer(r"^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", text, re.M):
        rows.append({
            "steps": int(m.group(1)),
            "sim": float(m.group(2)),
            "time": float(m.group(3)),
        })
    return rows


def parse_prompt_sims(log_path: Path):
    """从拓展1日志抽 提示词/相似度/耗时"""
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for m in re.finditer(r"^\s{2}(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+\(.+?\)\s*$",
                         text, re.M):
        prompt = m.group(1).strip()
        if len(prompt) < 5 or prompt.startswith("提示词"):
            continue
        rows.append({
            "prompt": prompt,
            "sim": float(m.group(2)),
            "time": float(m.group(3)),
        })
    return rows


def parse_recall(log_path: Path):
    """从子任务2日志抽 Recall"""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for m in re.finditer(r"文本→图像 Recall@(\d)\s*：(\d+)/(\d+)\s*=\s*([\d.]+)%",
                         text):
        out[f"R@{m.group(1)}"] = {
            "hit": int(m.group(2)), "total": int(m.group(3)),
            "rate": float(m.group(4)),
        }
    return out


def parse_clip_compare(log_path: Path):
    """从拓展1日志抽三组命中率"""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for m in re.finditer(r"(原版CLIP \+ 中文描述|原版CLIP \+ 英文描述|中文CLIP \+ 中文描述)"
                         r"\s+命中率：(\d+)/(\d+)\s*=\s*([\d.]+)%", text):
        out[m.group(1)] = {
            "hit": int(m.group(2)), "total": int(m.group(3)),
            "rate": float(m.group(4)),
        }
    return out


def parse_whisper(log_path: Path):
    """从 Whisper 日志抽转录结果"""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    m = re.search(r"识别结果：(.+)", text)
    if m:
        out["text"] = m.group(1).strip()
    m = re.search(r"转录耗时：([\d.]+) 秒", text)
    if m:
        out["time"] = float(m.group(1))
    m = re.search(r"最长公共子串长度：(\d+)/(\d+)", text)
    if m:
        out["lcs"] = f"{m.group(1)}/{m.group(2)}"
    return out


def parse_tts_compare(log_path: Path):
    """从 --compare 日志里抽两平台的时长与 RMS 响度

    日志行形如：
      日常对话                   3.28s         4.25s          -25.5          -18.0
    """
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for m in re.finditer(
            r"^\s+(\S+)\s+([\d.]+)s\s+([\d.]+)s\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$",
            text, re.M):
        rows.append({
            "tag": m.group(1),
            "dur_c": float(m.group(2)),
            "dur_m": float(m.group(3)),
            "rms_c": float(m.group(4)),
            "rms_m": float(m.group(5)),
        })
    return rows


def parse_tts_cer(log_path: Path):
    """从 TTS 质量评估日志里抽 CER（ChatTTS 与 minimax 两个平台）

    注意：必须先定位「【类别】」所在行，再在同一段内找 CER，
    不能用 re.S 让 .+? 跨行匹配（会把上一段的数字也吞进来）。
    """
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    cur_tag = None
    for ln in lines:
        m = re.match(r"\s*【(.+?)】", ln)
        if m:
            cur_tag = m.group(1)
            continue
        if cur_tag is None:
            continue
        mc = re.search(r"ChatTTS CER = ([\d.]+)%", ln)
        mm = re.search(r"minimax CER = ([\d.]+)%", ln)
        if mc:
            rows.append({"tag": cur_tag,
                         "cer": float(mc.group(1)),
                         "cer_m": float(mm.group(1)) if mm else None})
            cur_tag = None
    return rows


def main():
    ap = argparse.ArgumentParser(description="生成实践结果汇总表")
    ap.add_argument("--out", default="结果汇总.md")
    args = ap.parse_args()

    banner("实践结果汇总")

    md = []
    md.append(f"# 任务3 实测结果汇总\n")
    md.append(f"：　：{STUDENT_ID}　"
              f"设备：NVIDIA GeForce RTX 4060 Laptop GPU\n")

    # --- 任务一 ---
    section("任务一 · CLIP")
    recall = parse_recall(LOG_DIR / "任务一_子任务2_MN匹配.log")
    cmp_res = parse_clip_compare(LOG_DIR / "任务一_拓展1_原版CLIP对比.log")
    if recall:
        md.append("### 1.2 子任务2 检索指标（M=5 条文本 × N=6 张图片）\n")
        md.append("| 指标 | 命中 | 总数 | 比率 |")
        md.append("|------|------|------|------|")
        for k, v in sorted(recall.items()):
            md.append(f"| 文本→图像 {k} | {v['hit']} | {v['total']} | {v['rate']:.1f}% |")
        md.append("")
    if cmp_res:
        md.append("### 1.4 拓展1 模型对比命中率\n")
        md.append("| 模型 + 描述语言 | 命中 | 总数 | 命中率 |")
        md.append("|----------------|------|------|--------|")
        for k, v in cmp_res.items():
            md.append(f"| {k} | {v['hit']} | {v['total']} | {v['rate']:.1f}% |")
        md.append("")

    # --- 任务二 ---
    section("任务二 · Stable Diffusion")
    steps = parse_task2_steps(LOG_DIR / "任务二_子任务2_推理步数.log")
    if steps:
        md.append("### 2.2 子任务2 推理步数与耗时\n")
        md.append("| num_inference_steps | 总耗时(s) | 每步耗时(ms) | 相对耗时 |")
        md.append("|-------------------|-----------|--------------|----------|")
        for r in steps:
            md.append(f"| {r['steps']} | {r['total']:.2f} | {r['per_step_ms']:.1f} | {r['ratio']} |")
        md.append("")
    ev = parse_task3_eval(LOG_DIR / "任务二_拓展1_CLIP评估.log")
    if ev:
        md.append("### 2.4 拓展1 不同步数的 CLIP 相似度（实践B）\n")
        md.append("| 步数 | CLIP余弦相似度 | 耗时(s) |")
        md.append("|------|---------------|---------|")
        for r in ev:
            md.append(f"| {r['steps']} | {r['sim']:.4f} | {r['time']:.2f} |")
        md.append("")
    ps = parse_prompt_sims(LOG_DIR / "任务二_拓展1_CLIP评估.log")
    if ps:
        md.append("### 2.5 不同提示词的 CLIP 相似度（实践A）\n")
        md.append("| 提示词 | CLIP余弦相似度 | 耗时(s) |")
        md.append("|------|---------------|---------|")
        for r in ps:
            md.append(f"| {r['prompt'][:56]} | {r['sim']:.4f} | {r['time']:.2f} |")
        md.append("")

    # --- 任务三 ---
    section("任务三 · Whisper")
    w = parse_whisper(LOG_DIR / "任务三_Whisper基础测试.log")
    if w:
        md.append("### 3.1 Whisper 转录结果\n")
        if "text" in w:
            md.append(f"- 转录文本：`{w['text']}`")
        if "time" in w:
            md.append(f"- 转录耗时：{w['time']:.2f} 秒")
        if "lcs" in w:
            md.append(f"- 数字串与正确的最长公共子串：{w['lcs']}")
        md.append("")

    tts = parse_tts_cer(LOG_DIR / "任务三_TTS质量评估.log")
    cmp_tts = parse_tts_compare(LOG_DIR / "任务三_双平台指标对比.log")
    if cmp_tts:
        md.append("### 3.3 双平台客观指标对比（时长 / 响度）\n")
        md.append("| 文本类别 | ChatTTS 时长(s) | minimax 时长(s) | ChatTTS RMS(dB) | minimax RMS(dB) |")
        md.append("|----------|-----------------|-----------------|-----------------|-----------------|")
        for r in cmp_tts:
            md.append(f"| {r['tag']} | {r['dur_c']:.2f} | {r['dur_m']:.2f} | "
                      f"{r['rms_c']:.1f} | {r['rms_m']:.1f} |")
        md.append("")
        md.append("> ChatTTS 采样率 24kHz、minimax 32kHz，均为单声道 16bit。")
        md.append("> minimax 三组 RMS 都比 ChatTTS 高约 2~7dB，说明它默认做了响度归一化，")
        md.append("> ChatTTS 保留了原始动态范围。时长上 minimax 普遍更长，语速更慢。")
        md.append("")
    if tts:
        md.append("### 3.4 ChatTTS 与 minimax 合成质量对比（Whisper 反验 CER）\n")
        md.append("| 文本类别 | ChatTTS CER | minimax CER |")
        md.append("|----------|-------------|-------------|")
        for r in tts:
            # 日志里的 CER 是百分数值（25.00 表示 25%），
            # 直接用 :.2% 会再乘 100 变成 2500.00%，要除掉
            c = f"{r['cer'] / 100:.2%}"
            m = f"{r['cer_m'] / 100:.2%}" if r.get("cer_m") is not None else "—"
            md.append(f"| {r['tag']} | {c} | {m} |")
        # 平均值
        if all(r.get("cer_m") is not None for r in tts):
            ac = sum(r["cer"] for r in tts) / len(tts) / 100
            am = sum(r["cer_m"] for r in tts) / len(tts) / 100
            md.append(f"| **平均** | **{ac:.2%}** | **{am:.2%}** |")
        md.append("")
        md.append("> CER 由 Whisper 对合成音频重新识别后与原文本比对得出（字符错误率）。")
        md.append("> 越低说明合成语音越容易被 ASR 正确还原，即音质清晰、发音准确。")
        md.append("> 两个平台使用同一段文本、同一 ASR 模型，可比。")
        md.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(md), encoding="utf-8")
    section("已生成")
    print(f"  {out_path.resolve()}")
    print(f"  行数：{len(md)}")
    print()
    print("  说明：表里的数字全部来自 logs/ 下各脚本的真实运行输出，可直接引用。")
    print("  如果某张表是空的，说明对应脚本还没跑过 —— 先运行 run_all.bat。")

    footer("汇总完成")


if __name__ == "__main__":
    main()
