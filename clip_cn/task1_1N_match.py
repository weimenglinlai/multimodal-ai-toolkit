# -*- coding: utf-8 -*-
"""
任务一 · 子任务1：1 条中文语句与 N 张图片的匹配
===============================================
对应参考文档「（一）4．实践任务（1）子任务1」

需求：自行编写代码，实现 1 条中文语句与 N 张图片的匹配，输出匹配结果信息。
做法：对 N 张图片分别计算与该语句的余弦相似度，按相似度降序排名。

⚠️ 指标口径：真实余弦相似度 = logits_per_image / logit_scale（实测 logit_scale≈100）。
   直接用 logits_per_image 当相似度是错的（数值 20~35）。

运行：
    python clip_cn/task1_1N_match.py
    python clip_cn/task1_1N_match.py --query "一只橘猫趴在木质窗台上"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from common.clip_utils import clip_similarity, softmax_over_images
from common.info import CLIP_CN_DIR, PIC_DIR, banner, footer, section, device_info

DEFAULT_QUERY = "一只蜜蜂停在粉色花朵上"


def collect_images(folder: Path):
    """收集文件夹下所有图片"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser(description="1-N 文图匹配")
    ap.add_argument("--query", default=DEFAULT_QUERY, help="查询用的中文语句")
    ap.add_argument("--pic-dir", default=str(PIC_DIR), help="图片目录")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    banner("任务一 · 子任务1：1 条中文语句 与 N 张图片 的匹配")
    print(f"  模型目录：{CLIP_CN_DIR}")
    print(f"  查询语句：{args.query}")
    print(f"  图片目录：{args.pic_dir}")

    pic_dir = Path(args.pic_dir)
    images = collect_images(pic_dir)
    if not images:
        print(f"  [错误] 目录 {pic_dir} 下没有图片，请先运行 scripts/download_test_images.py")
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    section("加载模型")
    print(f"  推理设备：{device_info() if device.type == 'cuda' else 'CPU'}")
    print(f"  待匹配图片数 N = {len(images)}")

    model = ChineseCLIPModel.from_pretrained(str(CLIP_CN_DIR)).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(str(CLIP_CN_DIR))

    section("提取特征并计算相似度")
    pil_images = [Image.open(p).convert("RGB") for p in images]
    # sim: (M=1, N)；logits: (N, M=1)
    sim, logits = clip_similarity(model, processor, [args.query], pil_images, device)
    scores = sim[0]  # (N,) 每张图与该语句的余弦相似度
    # 1-N 场景：只有 1 条语句，概率分布应落在 N 张图片上
    probs = softmax_over_images(sim, 0)  # (N,)

    print(f"  原始 logits_per_image = {logits[:, 0].numpy().round(4).tolist()}")
    print(f"  还原后的余弦相似度    = {scores.numpy().round(4).tolist()}")
    print("  说明：余弦相似度取值范围 -1~1，越大表示图文语义越对齐。")

    section("匹配结果（按余弦相似度降序）")
    ranked = sorted(range(len(images)), key=lambda i: scores[i].item(), reverse=True)
    print(f"  查询语句：\"{args.query}\"")
    print(f"  {'排名':<8}{'图片文件':<26}{'余弦相似度':>14}{'softmax概率':>14}")
    for rank, i in enumerate(ranked, 1):
        print(f"  {rank:<10}{images[i].name:<28}{scores[i].item():>14.4f}{probs[i].item():>14.4f}")

    best = ranked[0]
    print()
    print(f"  >>> 最匹配图片：{images[best].name}")
    print(f"      余弦相似度 {scores[best].item():.4f}，"
          f"领先第2名 {scores[best].item() - scores[ranked[1]].item():.4f}")

    section("分析提示")
    print("  1. Top-1 是否与图片实际内容一致？→ 一致说明中文CLIP 的中文对齐能力良好。")
    print("  2. Top-1 与 Top-2 的相似度差值多大？→ 差值大说明区分度强、置信度高。")
    print("  3. 各候选之间相似度的绝对值普遍偏低（约 0.2~0.3）是否正常？")
    print("     → 正常。CLIP 是对比学习模型，绝对值没有语义，")
    print("       重要的是同一批候选内部的相对排序，而不是数值大小。")

    footer("子任务1 完成")


if __name__ == "__main__":
    main()
