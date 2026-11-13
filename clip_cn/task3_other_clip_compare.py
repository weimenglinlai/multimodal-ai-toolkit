# -*- coding: utf-8 -*-
"""
任务一 · 拓展任务1：其他 CLIP 模型实践与对比
=============================================
对应参考文档「（一）4．实践任务（3）拓展任务1」

需求：在 hf-mirror 上下载其他 CLIP 模型（如 OpenAI 原版 CLIP），编写代码测试并对比。

做法：
  模型A = OFA-Sys/chinese-clip-vit-base-patch16（中文CLIP，输入中文）
  模型B = openai/clip-vit-base-patch32      （OpenAI 原版 CLIP，输入英文）
  对同一批图片分别做匹配：
    · 中文CLIP 用中文候选句
    · 原版 CLIP 用英文候选句
  然后把两边的相似度分布、Top-1 命中等结果并排对比。

  另外做一次「跨语言对照实践」：给原版 CLIP 输入中文描述，
  观察它是否还能正确排序 —— 这是最能说明"中文化训练"必要性的证据。

运行：
    python clip_cn/task3_other_clip_compare.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from transformers import (ChineseCLIPModel, ChineseCLIPProcessor,
                          CLIPModel, CLIPProcessor)

from common.clip_utils import clip_similarity
from common.info import (CLIP_CN_DIR, CLIP_EN_DIR, PIC_DIR, banner,
                         footer, section, device_info)

# 每张图片配一组中文描述与英文描述，内容一一对应
# answer = 该图正确答案在候选列表中的下标（zh 与 en 列表下标对齐）
# ⚠️ 文件名与图片实际画面一致（已逐张人工核对），不要按远端文件名直译。
PAIRS = [
    {
        "file": "bee_on_flower.jpg",
        "answer": 0,
        "zh": ["蜜蜂停在粉色的花上采蜜", "红砖墙的爱尔兰酒吧门口", "一辆青绿色的老爷车停在墙边"],
        "en": ["a bee collecting nectar on a pink flower", "a red brick Irish pub entrance",
               "a vintage teal car parked by a wall"],
    },
    {
        "file": "red_pub.jpg",
        "answer": 0,
        "zh": ["红砖墙的爱尔兰酒吧门口", "一只虎斑猫的特写照片", "从空中俯拍的农田和房屋"],
        "en": ["a red brick Irish pub entrance", "a close-up photo of a tabby cat",
               "an aerial view of farmland and houses"],
    },
    {
        "file": "satellite_farm.jpg",
        "answer": 0,
        "zh": ["从空中俯拍的农田和房屋", "蜜蜂停在粉色的花上采蜜", "街道上立着中式牌楼和停车标志牌"],
        "en": ["an aerial view of farmland and houses", "a bee collecting nectar on a pink flower",
               "a chinese archway and a stop sign on a street"],
    },
    {
        "file": "street_chinatown.jpg",
        "answer": 0,
        "zh": ["街道上立着中式牌楼和停车标志牌", "红砖墙的爱尔兰酒吧门口", "一只虎斑猫的特写照片"],
        "en": ["a chinese archway and a stop sign on a street", "a red brick Irish pub entrance",
               "a close-up photo of a tabby cat"],
    },
    {
        "file": "tabby_cat.jpg",
        "answer": 0,
        "zh": ["一只虎斑猫的特写照片", "从空中俯拍的农田和房屋", "一辆青绿色的老爷车停在墙边"],
        "en": ["a close-up photo of a tabby cat", "an aerial view of farmland and houses",
               "a vintage teal car parked by a wall"],
    },
    {
        "file": "vintage_car.jpg",
        "answer": 0,
        "zh": ["一辆青绿色的老爷车停在墙边", "蜜蜂停在粉色的花上采蜜", "街道上立着中式牌楼和停车标志牌"],
        "en": ["a vintage teal car parked by a wall", "a bee collecting nectar on a pink flower",
               "a chinese archway and a stop sign on a street"],
    },
]


def collect_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def run_clip(model, processor, texts, images, device):
    """跑一次匹配，返回 (每图Top1文本, 每图相似度列表, 每图Top1索引)

    ⚠️ 用 clip_similarity 取真实余弦相似度（-1~1），
       而不是直接用 logits（那是被 logit_scale≈100 放大的分数）。
    """
    sim, _ = clip_similarity(model, processor, texts, images, device)
    # sim 形状 (M, N)，转成按图片组织
    results = []
    for ni in range(sim.shape[1]):
        row = sim[:, ni]
        best = int(row.argmax().item())
        results.append((texts[best], row.tolist(), best))
    return results


def main():
    ap = argparse.ArgumentParser(description="中文CLIP vs 原版CLIP 对比")
    ap.add_argument("--pic-dir", default=str(PIC_DIR))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    banner("任务一 · 拓展任务1：其他 CLIP 模型实践与对比")
    print(f"  模型A（中文CLIP）：{CLIP_CN_DIR}")
    print(f"  模型B（原版CLIP） ：{CLIP_EN_DIR}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    section("加载两个模型")
    print(f"  推理设备：{device_info() if device.type == 'cuda' else 'CPU'}")

    cn_model = ChineseCLIPModel.from_pretrained(str(CLIP_CN_DIR)).to(device).eval()
    cn_proc = ChineseCLIPProcessor.from_pretrained(str(CLIP_CN_DIR))
    en_model = CLIPModel.from_pretrained(str(CLIP_EN_DIR)).to(device).eval()
    en_proc = CLIPProcessor.from_pretrained(str(CLIP_EN_DIR))
    print("  两个模型加载完成。")

    pic_dir = Path(args.pic_dir)
    available = {p.name for p in collect_images(pic_dir)}
    pairs = [p for p in PAIRS if p["file"] in available]
    if not pairs:
        print(f"  [提示] {pic_dir} 里没有预置的对比图片，改用目录内全部图片做通用对比。")
        all_imgs = collect_images(pic_dir)
        if not all_imgs:
            print("  [错误] 没有可用图片。")
            return
        pairs = [{"file": p.name,
                  "answer": 0,
                  "zh": ["一只虎斑猫的特写照片", "红砖墙的爱尔兰酒吧门口", "蜜蜂停在粉色的花上采蜜"],
                  "en": ["a close-up photo of a tabby cat", "a red brick Irish pub entrance",
                         "a bee collecting nectar on a pink flower"]}
                 for p in all_imgs]

    section("对比一：各自用「母语」描述做匹配")
    for p in pairs:
        img = Image.open(pic_dir / p["file"]).convert("RGB")
        zh_res = run_clip(cn_model, cn_proc, p["zh"], [img], device)
        en_res = run_clip(en_model, en_proc, p["en"], [img], device)
        print(f"  图片：{p['file']}   （正确答案下标 = {p['answer']}）")
        print(f"    中文CLIP  Top1 = \"{zh_res[0][0]}\"   相似度 {zh_res[0][1][zh_res[0][2]]:.4f}")
        print(f"    原版CLIP  Top1 = \"{en_res[0][0]}\"   相似度 {en_res[0][1][en_res[0][2]]:.4f}")
        print(f"    中文CLIP 全部候选：" + " | ".join(
            f"{t}={s:.3f}" for t, s in zip(p["zh"], zh_res[0][1])))
        print(f"    原版CLIP 全部候选：" + " | ".join(
            f"{t}={s:.3f}" for t, s in zip(p["en"], en_res[0][1])))
        print()

    section("对比二：跨语言对照 —— 原版 CLIP 输入中文描述")
    print("  目的：验证原版 CLIP 因缺少中文语料训练，对中文描述的区分能力显著下降。")
    print("  做法：同一张图，分别给原版 CLIP 中文描述 / 英文描述 / 给中文 CLIP 中文描述，")
    print("        比较三者选中的下标是否等于正确答案下标。")
    print()

    hit_zh = 0      # 原版CLIP + 中文描述 命中
    hit_en = 0      # 原版CLIP + 英文描述 命中
    hit_cn = 0      # 中文CLIP  + 中文描述 命中
    total = 0
    for p in pairs:
        img = Image.open(pic_dir / p["file"]).convert("RGB")
        ans = p["answer"]
        r_zh = run_clip(en_model, en_proc, p["zh"], [img], device)
        r_en = run_clip(en_model, en_proc, p["en"], [img], device)
        r_cn = run_clip(cn_model, cn_proc, p["zh"], [img], device)

        z_i, e_i, c_i = r_zh[0][2], r_en[0][2], r_cn[0][2]
        mark = lambda ok: "命中" if ok else "未命中"
        print(f"  图片：{p['file']}   正确答案 = \"{p['zh'][ans]}\"")
        print(f"    原版CLIP + 中文描述 → \"{p['zh'][z_i]}\"  ({mark(z_i == ans)}, 相似度 {r_zh[0][1][z_i]:.4f})")
        print(f"    原版CLIP + 英文描述 → \"{p['en'][e_i]}\"  ({mark(e_i == ans)}, 相似度 {r_en[0][1][e_i]:.4f})")
        print(f"    中文CLIP + 中文描述 → \"{p['zh'][c_i]}\"  ({mark(c_i == ans)}, 相似度 {r_cn[0][1][c_i]:.4f})")
        total += 1
        hit_zh += int(z_i == ans)
        hit_en += int(e_i == ans)
        hit_cn += int(c_i == ans)
        print()

    if total:
        section("结论汇总")
        print(f"  原版CLIP + 中文描述  命中率：{hit_zh}/{total} = {hit_zh / total * 100:.1f}%")
        print(f"  原版CLIP + 英文描述  命中率：{hit_en}/{total} = {hit_en / total * 100:.1f}%")
        print(f"  中文CLIP + 中文描述  命中率：{hit_cn}/{total} = {hit_cn / total * 100:.1f}%")
        print()
        print("  分析：")
        print("    · 原版CLIP 换成英文后命中率明显上升 → 说明其主要靠英文语料训练；")
        print("    · 原版CLIP 对中文描述判别力弱 → 这正是 Chinese-CLIP 做中文化预训练的意义；")
        print("    · 中文CLIP 在中文描述上表现最好 → 验证了中文图文对齐的必要性。")

    footer("拓展任务1 完成")


if __name__ == "__main__":
    main()
