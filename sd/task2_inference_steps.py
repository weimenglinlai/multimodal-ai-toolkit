# -*- coding: utf-8 -*-
"""
任务二 · 子任务2：推理步数的权衡 —— 速度与质量的平衡
====================================================
对应参考文档「（二）4．实践任务（2）子任务2」

目的：探究 num_inference_steps 对生成时间和图像质量的影响。
做法：固定同一个提示词，把 num_inference_steps 设为 10/20/30/50/100，
      用 time 模块计时，每组保存 1 张图，并输出「耗时 / 步数」的比值。

运行：
    python sd/task2_inference_steps.py
    python sd/task2_inference_steps.py --prompt "A futuristic city at night, neon lights, cyberpunk style"
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import DiffusionPipeline

from common.info import SD_DIR, SD_OUT_DIR, banner, footer, section, device_info

DEFAULT_PROMPT = "A futuristic city at night, neon lights, cyberpunk style"
DEFAULT_STEPS = [10, 20, 30, 50, 100]


def main():
    ap = argparse.ArgumentParser(description="推理步数权衡实践")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS, help="要测试的步数列表")
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    banner("任务二 · 子任务2：推理步数的权衡 —— 速度与质量的平衡")
    print(f"  模型目录：{SD_DIR}")
    print(f"  固定提示词：{args.prompt}")
    print(f"  测试步数  ：{args.steps}")
    print(f"  固定参数  ：CFG={args.guidance}  seed={args.seed}")

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

    section("逐组计时生成")
    results = []
    for s in args.steps:
        print(f"\n  → num_inference_steps = {s}")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        # 预热一次，把首次 CUDA 初始化开销排除在计时外
        if s == args.steps[0] and use_gpu:
            _ = pipe(args.prompt, num_inference_steps=2,
                     guidance_scale=args.guidance, generator=generator).images[0]
            generator = torch.Generator(device=device).manual_seed(args.seed)
            print("    (已完成一次 2 步预热，排除 CUDA 初始化开销)")
        t0 = time.time()
        image = pipe(
            args.prompt,
            num_inference_steps=s,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        elapsed = time.time() - t0
        out_path = SD_OUT_DIR / f"task2_steps{s:03d}.png"
        image.save(out_path)
        print(f"    生成耗时：{elapsed:.2f} 秒    已保存：{out_path.name}")
        results.append((s, elapsed, out_path))

    section("耗时汇总")
    print(f"  {'步数':>6}{'总耗时(s)':>14}{'每步耗时(ms)':>16}{'相对10步耗时':>16}")
    base = results[0][1]
    for s, elapsed, _ in results:
        print(f"  {s:>8}{elapsed:>14.2f}{elapsed / s * 1000:>16.1f}{elapsed / base:>15.2f}x")

    section("分析提示")
    print("  1. 每步耗时是否基本恒定？→ 若恒定，说明总耗时与步数近似线性。")
    print("  2. 观察 10 / 20 / 30 步：建筑结构与光影是否已明显成型？")
    print("  3. 观察 50 / 100 步：细节纹理提升是否明显？是否只是耗时翻倍？")
    print("  4. 实际应用中建议：在质量提升的边际收益拐点处选步数，")
    print("     通常 20~30 步即可满足绝大多数需求，再往上收益递减。")

    footer("子任务2 完成")


if __name__ == "__main__":
    main()
