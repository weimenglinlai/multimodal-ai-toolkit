# -*- coding: utf-8 -*-
"""
任务二 · 子任务1：提示词工程探索 —— 关键词对生成结果的影响
==========================================================
对应参考文档「（二）4．实践任务（1）子任务1」

目的：理解提示词的描述精度、关键词顺序对生成图像的影响。
做法：固定 num_inference_steps=30、seed 等参数，仅修改 prompt，
      设计 3 组（可扩展）提示词各生成 1 张图并保存，最后拼成对比图。

三组提示词（参考文档示例）：
  基础版      : "A cat"
  细节版      : "A white cat with blue eyes, sitting on a wooden table, watercolor style"
  关键词顺序版: "Watercolor style, a wooden table with a white cat sitting on it, blue eyes"

运行：
    python sd/task1_prompt_engineering.py
    python sd/task1_prompt_engineering.py --steps 30 --seed 42
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import DiffusionPipeline

from common.info import SD_DIR, SD_OUT_DIR, banner, footer, section, device_info

# (标签, 提示词, 设计意图)
PROMPT_GROUPS = [
    ("base", "A cat", "基础版：只给主体，信息量最少"),
    ("detail", "A white cat with blue eyes, sitting on a wooden table, watercolor style",
     "细节版：补充毛色、眼睛、姿态、场景、画风"),
    ("reorder", "Watercolor style, a wooden table with a white cat sitting on it, blue eyes",
     "关键词顺序版：与细节版语义完全相同，只把画风提到最前"),
]


def main():
    ap = argparse.ArgumentParser(description="提示词工程对比")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    banner("任务二 · 子任务1：提示词工程探索 —— 关键词对生成结果的影响")
    print(f"  模型目录：{SD_DIR}")
    print(f"  固定参数：num_inference_steps={args.steps}  CFG={args.guidance}  seed={args.seed}")
    print("  对比维度：提示词信息量 + 关键词顺序")

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    section("加载模型")
    print(f"  推理设备：{device_info() if use_gpu else 'CPU'}   dtype={dtype}")
    pipe = DiffusionPipeline.from_pretrained(
        str(SD_DIR), torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("  模型加载完成。")

    section("逐组生成")
    results = []
    for tag, prompt, intent in PROMPT_GROUPS:
        print(f"\n  【{tag}】{intent}")
        print(f"    提示词：{prompt}")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        t0 = time.time()
        image = pipe(
            prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        elapsed = time.time() - t0
        out_path = SD_OUT_DIR / f"task1_{tag}.png"
        image.save(out_path)
        print(f"    生成耗时：{elapsed:.2f} 秒    已保存：{out_path.name}")
        results.append((tag, prompt, intent, elapsed, out_path))

    section("汇总")
    print(f"  {'组别':<10}{'提示词':<62}{'耗时(s)':>10}")
    for tag, prompt, intent, elapsed, out_path in results:
        print(f"  {tag:<12}{prompt:<62}{elapsed:>10.2f}")

    section("对比观察要点（供报告分析参考）")
    print("  1. 基础版是否只出现猫的轮廓？毛色/眼睛/场景等细节词是否被体现？")
    print("  2. 细节版是否同时满足 毛色、眼睛颜色、姿态、桌面、水彩风 五个约束？")
    print("  3. 细节版与顺序版语义相同，若画面差异明显，说明 CLIP 文本编码器")
    print("     对提示词起始位置的词赋予更高权重（首词偏置 / 主体突出效应）。")
    print("  4. 三组共用同一 seed 与步数，因此差异只可能来自提示词本身。")

    footer("子任务1 完成")


if __name__ == "__main__":
    main()
