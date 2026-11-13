# -*- coding: utf-8 -*-
"""
任务二 · 基于 Stable Diffusion 的文本-图像生成 —— 基础测试程序
============================================================
对应参考文档「（二）3．代码编辑和运行（1）测试代码编写」

功能：用 segmind/tiny-sd 根据一段英文提示词生成 1 张图片。

与参考文档示例的差异（已在报告中说明）：
  参考文档用 CPU + float32，本机有 RTX 4060，因此默认走 GPU + float16，
  生成速度提升约一个数量级。加 --cpu 可切回参考文档的 CPU 路线复现。

运行：
    python sd/test_tiny_gen_one.py
    python sd/test_tiny_gen_one.py --prompt "a futuristic city at night" --steps 30
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from diffusers import DiffusionPipeline

from common.info import SD_DIR, SD_OUT_DIR, banner, footer, section, device_info

DEFAULT_PROMPT = "A serene mountain landscape with a crystal lake, snow-capped peaks, and a clear blue sky, digital art"


def main():
    ap = argparse.ArgumentParser(description="tiny-sd 文生图基础测试")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sd_test_basic.png")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU + float32")
    args = ap.parse_args()

    banner("任务二 · Stable Diffusion 文生图（基础测试）")
    print(f"  模型目录：{SD_DIR}")
    print(f"  提示词  ：{args.prompt}")
    print(f"  推理步数：{args.steps}   CFG={args.guidance}   seed={args.seed}")

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    dtype = torch.float16 if use_gpu else torch.float32

    section("加载模型")
    print(f"  推理设备：{device_info() if use_gpu else 'CPU'}")
    print(f"  数据类型：{dtype}")
    pipe = DiffusionPipeline.from_pretrained(
        str(SD_DIR),
        torch_dtype=dtype,
        safety_checker=None,          # tiny-sd 无安全检查器，显式关闭避免警告
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("  模型加载完成。")

    section("生成图像")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t0 = time.time()
    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]
    elapsed = time.time() - t0

    out_path = SD_OUT_DIR / args.out
    image.save(out_path)
    print(f"  生成耗时：{elapsed:.2f} 秒")
    print(f"  图片尺寸：{image.size}")
    print(f"  已保存  ：{out_path}")

    footer("基础测试完成")


if __name__ == "__main__":
    main()
