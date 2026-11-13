# -*- coding: utf-8 -*-
"""
把生成的图片拼成对比图，方便截图
================================
说明要求子任务1 提交 3 张图、子任务2 提交 5 张图，
但如果每张都单独截图会非常繁琐。本脚本把多张生成图拼成一张带标注的对比图，
截一次图就能同时展示全部结果。

运行：
    python scripts/make_contact_sheet.py
    python scripts/make_contact_sheet.py --group task1 --cols 3
    python scripts/make_contact_sheet.py --group task2 --cols 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from common.info import SD_OUT_DIR, banner, footer, section

# (分组名, 文件匹配前缀, 说明)
GROUPS = {
    "task1": ("task1_", "子任务1 · 提示词工程对比"),
    "task2": ("task2_steps", "子任务2 · 推理步数对比"),
    "eval": ("eval_", "拓展任务1 · CLIP 评估用图"),
}

FONT_CANDIDATES = [
    "./data",
    "./data",
    "./data",
]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def collect(prefix: str):
    files = sorted([p for p in SD_OUT_DIR.glob("*.png") if p.name.startswith(prefix)])
    return files


def build_sheet(files, out_path: Path, cols: int, title: str, labels=None):
    if not files:
        print(f"  [跳过] 没有匹配 {prefix} 的图片")
        return None

    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        # 统一缩放到 320x320
        im.thumbnail((320, 320), Image.LANCZOS)
        imgs.append(im)

    cell_w, cell_h = 340, 380
    label_h = 34
    title_h = 78          # 标题 + 副标题两行，留足高度避免重叠
    rows = (len(imgs) + cols - 1) // cols

    sheet = Image.new("RGB", (cols * cell_w, title_h + rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    title_font = load_font(26)
    label_font = load_font(17)
    small_font = load_font(15)

    # 标题（主标题 + 副标题： + 实践名）
    draw.text((16, 12), title, fill="black", font=title_font)
    draw.text((16, 46), "：    ：    《AI》任务3",
              fill="black", font=small_font)
    draw.line([(0, title_h - 4), (sheet.width, title_h - 4)], fill="black", width=2)

    for idx, (im, f) in enumerate(zip(imgs, files)):
        r, c = divmod(idx, cols)
        cell_top = title_h + r * (cell_h + label_h)
        x = c * cell_w + (cell_w - im.width) // 2
        y = cell_top + 8
        sheet.paste(im, (x, y))
        draw.rectangle([c * cell_w + 4, cell_top + 4,
                        (c + 1) * cell_w - 6, cell_top + cell_h + 4],
                       outline="gray", width=1)
        # 标注文字放在边框下方，不与图片区域重叠
        label = labels[idx] if labels else f.stem
        draw.text((c * cell_w + 12, cell_top + cell_h + 12),
                  label, fill="black", font=label_font)

    sheet.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="拼接生成图对比图")
    ap.add_argument("--group", choices=list(GROUPS.keys()), default="task1")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prefix, title = GROUPS[args.group]
    banner(f"拼接对比图 · {title}")
    print(f"  匹配前缀：{prefix}")
    print(f"  输出目录：{SD_OUT_DIR}")

    section("收集图片")
    files = collect(prefix)
    for f in files:
        print(f"  {f.name}")

    labels = None
    if args.group == "task1":
        label_map = {
            "task1_base.png": "基础版: A cat",
            "task1_detail.png": "细节版: 毛色+眼睛+姿态+场景+画风",
            "task1_reorder.png": "顺序版: 画风前置，语义同细节版",
        }
        labels = [label_map.get(f.name, f.stem) for f in files]
    elif args.group == "task2":
        labels = [f.stem.replace("task2_steps", "步数 = ") for f in files]

    section("拼接")
    out = Path(args.out) if args.out else SD_OUT_DIR / f"sheet_{args.group}.png"
    result = build_sheet(files, out, args.cols, title, labels)
    if result:
        print(f"  已生成：{result}")
        print(f"  尺寸：{Image.open(result).size}")
        print()
        print("  截图建议：把这张对比图和终端输出并排放在一张截图里，")
        print("            终端部分要能看到「  /  」和耗时数据。")
    else:
        print("  未生成对比图。")

    footer("对比图生成完成")


if __name__ == "__main__":
    main()
