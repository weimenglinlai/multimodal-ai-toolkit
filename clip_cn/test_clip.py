# -*- coding: utf-8 -*-
"""
任务一 · 基于中文CLIP的文本-图像匹配 —— 基础测试程序
=================================================
对应参考文档「（一）3．代码编辑和运行（1）测试代码编写」

功能：给定 1 张图片 + 4 条候选中文描述，判断哪条描述最匹配图片内容。
模型：OFA-Sys/chinese-clip-vit-base-patch16（本地目录加载）

⚠️ 与参考文档示例的差异（已在报告中说明）：
  参考文档直接打印 `logits_per_image`，那是被 logit_scale(≈100) 放大过的
  对数分数（20~35），不是余弦相似度。本脚本额外用
  logits / logit_scale 还原出真正的余弦相似度（-1~1），
  并同时给出 softmax 概率，便于对比两种口径。

运行：
    python clip_cn/test_clip.py
    python clip_cn/test_clip.py --image clip_cn/pic/fireworks.jpg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from common.clip_utils import clip_similarity
from common.info import CLIP_CN_DIR, PIC_DIR, banner, footer, section, device_info

# 候选中文句子
# 参考文档原文用的是"大红灯笼/过新年放烟花/火烧云/夜幕降临"，
# 但那组句子对应的是参考文档自带的 festival.jpg。
# 本项目的测试图换成 bee_on_flower.jpg，故候选句也换成对应内容，
# 保证「图片 + 候选句」严格对应（否则模型输出无法解释）。
DEFAULT_TEXTS = [
    "蜜蜂停在粉色的花上采蜜",
    "红砖墙的爱尔兰酒吧门口",
    "一辆青绿色的老爷车停在墙边",
    "从空中俯拍的农田和房屋",
]


def main():
    ap = argparse.ArgumentParser(description="中文CLIP 图文匹配基础测试")
    ap.add_argument("--image", default=str(PIC_DIR / "bee_on_flower.jpg"), help="待测试图片路径")
    ap.add_argument("--texts", nargs="+", default=DEFAULT_TEXTS, help="候选中文描述")
    ap.add_argument("--cpu", action="store_true", help="强制用 CPU 推理")
    args = ap.parse_args()

    banner("任务一 · 中文CLIP 文本-图像匹配（基础测试）")
    print(f"  模型目录：{CLIP_CN_DIR}")
    print(f"  测试图片：{args.image}")
    print(f"  候选文本：{args.texts}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    section("加载模型")
    print(f"  推理设备：{device_info() if device.type == 'cuda' else 'CPU'}")

    model = ChineseCLIPModel.from_pretrained(str(CLIP_CN_DIR)).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(str(CLIP_CN_DIR))
    print(f"  模型加载完成。logit_scale = {model.logit_scale.exp().item():.2f}")

    section("计算图文相似度")
    image = Image.open(args.image).convert("RGB")
    # sim: (M, N) 这里 N=1；logits: (N, M)
    sim, logits = clip_similarity(model, processor, args.texts, [image], device)

    print(f"  原始 logits_per_image（缩放后的对数分数）= "
          f"{logits[0].numpy().round(4).tolist()}")
    print(f"  还原后的余弦相似度（logits / logit_scale）= "
          f"{sim[:, 0].numpy().round(4).tolist()}")

    probs = torch.softmax(logits[0], dim=0)
    print()
    print("  候选文本匹配度排序：")
    ranked = sorted(range(len(args.texts)), key=lambda i: sim[i, 0].item(), reverse=True)
    for rank, i in enumerate(ranked, 1):
        print(f"    {rank}. {args.texts[i]:<24s}  余弦相似度 {sim[i, 0].item():>8.4f}"
              f"   softmax概率 {probs[i].item():>8.4f}")

    best = ranked[0]
    print()
    print(f"  >>> 最匹配描述：\"{args.texts[best]}\"")
    print(f"      余弦相似度 {sim[best, 0].item():.4f}，softmax 概率 {probs[best].item():.4f}")

    footer("基础测试完成")


if __name__ == "__main__":
    main()
