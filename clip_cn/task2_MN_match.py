# -*- coding: utf-8 -*-
"""
任务一 · 子任务2：M 条中文语句与 N 张图片的匹配
===============================================
对应参考文档「（一）4．实践任务（2）子任务2」

需求：自行编写代码，实现 M 条中文语句与 N 张图片的匹配，输出匹配结果信息。
做法：一次前向得到 M×N 的完整相似度矩阵，然后做两个方向的分析：
      1) 图像→文本（每张图最匹配哪句话）—— 图文检索
      2) 文本→图像（每句话最匹配哪张图）—— 文图检索
      并给出整张矩阵的 Top-K 配对列表。

⚠️ 指标口径：真实余弦相似度 = logits_per_image / logit_scale（实测 logit_scale≈100）。
   同时输出 Recall@1 等可量化的检索指标。

运行：
    python clip_cn/task2_MN_match.py
    python clip_cn/task2_MN_match.py --texts "一只橘猫趴在木质窗台上" "城市里的一辆汽车"
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

# 默认 6 条中文语句，与 6 张测试图一一对应
DEFAULT_TEXTS = [
    "蜜蜂停在粉色的花上采蜜",
    "红砖墙的爱尔兰酒吧门口",
    "从空中俯拍的农田和房屋",
    "街道上立着中式牌楼和停车标志牌",
    "一只虎斑猫的特写照片",
    "一辆青绿色的老爷车停在墙边",
]

# 图片的"标准答案"（用于计算 Recall@1）：文本下标 → 图片文件名
# 文本顺序与图片文件名顺序故意不一致，用来验证模型做的是语义匹配而非位置对齐
GROUND_TRUTH = {
    0: "bee_on_flower.jpg",
    1: "red_pub.jpg",
    2: "satellite_farm.jpg",
    3: "street_chinatown.jpg",
    4: "tabby_cat.jpg",
    5: "vintage_car.jpg",
}


def collect_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser(description="M-N 文图匹配")
    ap.add_argument("--texts", nargs="+", default=DEFAULT_TEXTS, help="M 条中文语句")
    ap.add_argument("--pic-dir", default=str(PIC_DIR))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    banner("任务一 · 子任务2：M 条中文语句 与 N 张图片 的匹配")
    print(f"  模型目录：{CLIP_CN_DIR}")
    print(f"  图片目录：{args.pic_dir}")
    print(f"  文本数量 M = {len(args.texts)}")

    pic_dir = Path(args.pic_dir)
    images = collect_images(pic_dir)
    if not images:
        print(f"  [错误] 目录 {pic_dir} 下没有图片。")
        return
    print(f"  图片数量 N = {len(images)}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    section("加载模型")
    print(f"  推理设备：{device_info() if device.type == 'cuda' else 'CPU'}")

    model = ChineseCLIPModel.from_pretrained(str(CLIP_CN_DIR)).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(str(CLIP_CN_DIR))

    section("计算 M×N 相似度矩阵")
    pil_images = [Image.open(p).convert("RGB") for p in images]
    # sim: (M, N)  —— 行=文本，列=图片
    sim, logits = clip_similarity(model, processor, args.texts, pil_images, device)

    print("\n  【余弦相似度矩阵】行 = 文本，列 = 图片")
    header = "  " + " " * 30 + "".join(f"{p.name[:15]:>17}" for p in images)
    print(header)
    print("  " + "-" * (30 + 17 * len(images)))
    for ti, t in enumerate(args.texts):
        row = f"  {t[:28]:<30}" + "".join(f"{sim[ti][ni].item():>17.4f}" for ni in range(len(images)))
        print(row)

    section("方向一 · 图像 → 文本（每张图最匹配哪句话）")
    img2txt_hit = 0
    img2txt_total = 0
    for ni, p in enumerate(images):
        col = sim[:, ni]
        order = sorted(range(len(args.texts)), key=lambda i: col[i].item(), reverse=True)
        print(f"  {p.name}")
        for rank, ti in enumerate(order, 1):
            print(f"      {rank}. {args.texts[ti]:<28} 余弦相似度 {col[ti].item():.4f}")
        print()

    section("方向二 · 文本 → 图像（每句话最匹配哪张图）")
    txt2img_hit = 0
    txt2img_total = 0
    for ti, t in enumerate(args.texts):
        row = sim[ti]
        order = sorted(range(len(images)), key=lambda i: row[i].item(), reverse=True)
        top1 = images[order[0]].name
        expect = GROUND_TRUTH.get(ti)
        ok = (expect is not None and top1 == expect)
        print(f"  \"{t}\"")
        for rank, ni in enumerate(order, 1):
            print(f"      {rank}. {images[ni].name:<28} 余弦相似度 {row[ni].item():.4f}")
        if expect:
            print(f"      正确答案 = {expect}   →  {'✓ 命中' if ok else '✗ 未命中'}")
            txt2img_total += 1
            txt2img_hit += int(ok)
        print()

    section("全局 Top-5 配对（跨全部 M×N 组合排序）")
    flat = [(sim[ti][ni].item(), ti, ni) for ti in range(len(args.texts)) for ni in range(len(images))]
    flat.sort(reverse=True)
    for rank, (s, ti, ni) in enumerate(flat[:5], 1):
        print(f"  {rank}. 余弦相似度 {s:.4f}   \"{args.texts[ti]}\"  <->  {images[ni].name}")

    section("检索指标")
    if txt2img_total:
        r1 = txt2img_hit / txt2img_total
        print(f"  文本→图像 Recall@1 ：{txt2img_hit}/{txt2img_total} = {r1 * 100:.1f}%")
        # Recall@2
        hit2 = 0
        for ti, expect in GROUND_TRUTH.items():
            if ti >= len(args.texts):
                continue
            row = sim[ti]
            order = sorted(range(len(images)), key=lambda i: row[i].item(), reverse=True)
            if expect in [images[i].name for i in order[:2]]:
                hit2 += 1
        print(f"  文本→图像 Recall@2 ：{hit2}/{txt2img_total} = {hit2 / txt2img_total * 100:.1f}%")

    section("分析提示")
    print("  1. 文本→图像的 Recall@1 越高说明中文CLIP 的中文检索能力越强。")
    print("  2. 图像→文本方向通常更容易，因为一张图往往能同时匹配多个描述，")
    print("     而一句话的语义更具体、更「挑剔」。")
    print("  3. 观察矩阵对角线是否是最亮的一批格子 → 对角线突出说明对齐良好。")
    print("  4. 注意余弦相似度绝对值普遍只有 0.2~0.3，这是 CLIP 的正常现象，")
    print("     它的意义在相对排序而非绝对数值。")

    footer("子任务2 完成")


if __name__ == "__main__":
    main()
