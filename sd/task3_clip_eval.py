# -*- coding: utf-8 -*-
"""
任务二 · 拓展任务1：自动评估生成质量 —— 用 CLIP 计算提示词与图像的相似度
========================================================================
对应参考文档「（二）4．实践任务（3）拓展任务1」

原理：CLIP 把文本和图像映射到同一特征空间，两者特征向量做余弦相似度
      即可量化「生成图是否忠实于提示词」，值越接近 1 匹配度越高。

做法：同时加载 tiny-sd（生成）与 OpenAI 原版 CLIP（评估），
      对不同提示词 / 不同推理步数生成图，逐张计算与提示词的 CLIP 相似度，
      并与耗时一起输出，形成量化对比表。

注意：tiny-sd 的文本编码器基于英文 CLIP 词表，故评估用英文 CLIP +
      英文提示词，图文特征同源，相似度才具可比性。

运行：
    python sd/task3_clip_eval.py
    python sd/task3_clip_eval.py --mode prompt
    python sd/task3_clip_eval.py --mode steps
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from transformers import CLIPModel, CLIPProcessor

from common.clip_utils import cosine_matrix, get_image_features, get_text_features
from common.info import CLIP_EN_DIR, SD_DIR, SD_OUT_DIR, banner, footer, section, device_info

# 用于「不同提示词」对比的提示词组
PROMPT_GROUPS = [
    ("A cat", "信息量最少"),
    ("A white cat with blue eyes, sitting on a wooden table, watercolor style", "细节丰富"),
    ("Watercolor style, a wooden table with a white cat sitting on it, blue eyes", "同样语义·调整顺序"),
    ("A majestic snow leopard on a rocky mountain, 8k, highly detailed, cinematic lighting", "更复杂的组合描述"),
]

# 用于「不同推理步数」对比的提示词
STEPS_PROMPT = "A futuristic city at night, neon lights, cyberpunk style"
STEPS_LIST = [10, 20, 30, 50, 100]


class ClipScorer:
    """封装 CLIP 相似度计算

    ⚠️ 两个实测坑（transformers 5.12.1）：
      1. get_image_features/get_text_features 返回 Output 对象而非 Tensor，
         且特征未做 L2 归一化 → 必须取 pooler_output 再手动归一化；
      2. logits_per_image = logit_scale × cosine，不是相似度本身。
      这里统一用 common.clip_utils 处理，避免口径错误。
    """

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.model = CLIPModel.from_pretrained(str(CLIP_EN_DIR)).to(device=device, dtype=dtype).eval()
        self.processor = CLIPProcessor.from_pretrained(str(CLIP_EN_DIR))

    @torch.no_grad()
    def similarity(self, prompt: str, image: Image.Image) -> float:
        """返回图像与提示词的余弦相似度"""
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
        inputs = {k: (v.to(self.device, self.dtype) if v.dtype.is_floating_point
                      else v.to(self.device)) for k, v in inputs.items()}
        tfeat = get_text_features(
            self.model,
            inputs["input_ids"],
            inputs.get("attention_mask"),
        )
        ifeat = get_image_features(self.model, inputs["pixel_values"])
        sim = cosine_matrix(tfeat.float(), ifeat.float())
        return float(sim[0, 0].item())


def main():
    ap = argparse.ArgumentParser(description="CLIP 自动评估 SD 生成质量")
    ap.add_argument("--mode", choices=["prompt", "steps", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    banner("任务二 · 拓展任务1：用 CLIP 计算提示词与生成图的相似度")
    print(f"  生成模型：{SD_DIR}")
    print(f"  评估模型：{CLIP_EN_DIR}   (OpenAI 原版 CLIP)")
    print(f"  实践模式：{args.mode}")

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
    scorer = ClipScorer(device, dtype)
    print("  SD 与 CLIP 均加载完成。")

    all_rows = []

    if args.mode in ("prompt", "both"):
        section("实践A：不同提示词的 CLIP 相似度对比（steps=30 固定）")
        print(f"  {'提示词':<62}{'相似度':>10}{'耗时(s)':>10}")
        for prompt, note in PROMPT_GROUPS:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            t0 = time.time()
            image = pipe(prompt, num_inference_steps=30,
                         guidance_scale=args.guidance, generator=generator).images[0]
            elapsed = time.time() - t0
            sim = scorer.similarity(prompt, image)
            tag = f"eval_prompt_{len(all_rows):02d}.png"
            image.save(SD_OUT_DIR / tag)
            print(f"  {prompt[:60]:<62}{sim:>10.4f}{elapsed:>10.2f}   ({note})")
            all_rows.append(("prompt", prompt, sim, elapsed))

    if args.mode in ("steps", "both"):
        section("实践B：不同推理步数的 CLIP 相似度对比（提示词固定）")
        print(f"  提示词：{STEPS_PROMPT}")
        print(f"  {'步数':>6}{'相似度':>12}{'耗时(s)':>12}")
        for s in STEPS_LIST:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            t0 = time.time()
            image = pipe(STEPS_PROMPT, num_inference_steps=s,
                         guidance_scale=args.guidance, generator=generator).images[0]
            elapsed = time.time() - t0
            sim = scorer.similarity(STEPS_PROMPT, image)
            image.save(SD_OUT_DIR / f"eval_steps{s:03d}.png")
            print(f"  {s:>8}{sim:>12.4f}{elapsed:>12.2f}")
            all_rows.append(("steps", f"steps={s}", sim, elapsed))

    section("结果汇总")
    print(f"  {'维度':<10}{'对象':<60}{'CLIP相似度':>14}{'耗时(s)':>12}")
    for kind, key, sim, elapsed in all_rows:
        print(f"  {kind:<12}{key[:58]:<60}{sim:>14.4f}{elapsed:>12.2f}")

    section("分析提示")
    print("  1. 提示词越具体，CLIP 相似度是否越高？")
    print("     → 相似度高说明生成图更「扣题」，但相似度高 ≠ 人看着好看；")
    print("       CLIP 只衡量图文语义对齐，不衡量构图美感与伪影。")
    print("  2. 步数增加时相似度是否单调上升？")
    print("     → 相似度提升到一定程度后趋平，说明多余步数只增加耗时，")
    print("       对「扣题程度」的边际贡献迅速衰减。")
    print("  3. 主观感受与 CLIP 分数是否一致？")
    print("     → 若不一致，通常因为 CLIP 对「语义大致正确」的图都给出高分，")
    print("       区分不了细节伪影；这正是 CLIP 作为自动指标的局限。")

    footer("拓展任务1 完成")


if __name__ == "__main__":
    main()
