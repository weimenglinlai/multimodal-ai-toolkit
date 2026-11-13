# -*- coding: utf-8 -*-
"""
下载 CLIP 实践用的测试图片
==========================
参考文档里给的测试图（festival.jpg 等）托管在 ModelScope 的 CLIP 教程文档里，
是文档插图而不是可直接下载的独立文件；且 Wikimedia 等图床在当前网络下超时。

本脚本从 HuggingFace 官方 documentation-images 数据集（经 hf-mirror 加速）拉取
若干张内容差异明显的照片作为测试集，保证 1-N / M-N 匹配实践有区分度。

运行：
    python scripts/download_test_images.py
    python scripts/download_test_images.py --force
"""
import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.info import PIC_DIR, banner, footer, section

MIRROR = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/transformers/tasks"

# (保存名, 远端文件名, 说明)
# ⚠️ 保存名按图片「实际画面内容」命名（已逐张人工核对过），不要按远端文件名直译，
#    否则会出现"文件名叫烟花、实际是卫星图"这种错配，实践结论会完全跑偏。
IMAGES = [
    ("bee_on_flower.jpg",     "bee.JPG",                "粉色花丛中采蜜的蜜蜂"),
    ("red_pub.jpg",           "temple-bar-dublin-world-famous-irish-pub.jpg", "红砖墙的爱尔兰酒吧"),
    ("satellite_farm.jpg",    "satellite-dataset.png",  "卫星俯拍的农田与房屋"),
    ("street_chinatown.jpg",  "australia.jpg",          "街道上的中式牌楼与停车牌"),
    ("tabby_cat.jpg",         "cat.jpg",                "虎斑猫的特写"),
    ("vintage_car.jpg",       "car.jpg",                "青绿色老爷车"),
]


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    # 15 字节的 "Entry not found" 是 hf-mirror 对不存在文件的返回，必须拦掉
    if len(data) < 1024:
        print(f"      [失败] 内容过小({len(data)}B)，文件可能不存在：{url}")
        return False
    dest.write_bytes(data)
    print(f"      [OK] {dest.name}  {len(data) / 1024:.0f} KB")
    return True


def main():
    ap = argparse.ArgumentParser(description="下载 CLIP 测试图片")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    banner("下载 CLIP 实践测试图片")
    print(f"  保存目录：{PIC_DIR}")
    print(f"  图片来源：hf-mirror.com / huggingface/documentation-images")

    section("开始下载")
    ok, fail = 0, 0
    for name, remote, desc in IMAGES:
        dest = PIC_DIR / name
        if dest.exists() and dest.stat().st_size > 1024 and not args.force:
            print(f"      [跳过] {name} 已存在（{dest.stat().st_size / 1024:.0f} KB）")
            ok += 1
            continue
        url = f"{MIRROR}/{remote}"
        print(f"  -> {name}  ({desc})")
        t0 = time.time()
        try:
            if download(url, dest):
                ok += 1
                print(f"         用时 {time.time() - t0:.1f}s")
            else:
                fail += 1
        except Exception as e:
            print(f"      [失败] {type(e).__name__}: {e}")
            fail += 1

    section("结果")
    print(f"  成功 {ok} 张，失败 {fail} 张")
    for p in sorted(PIC_DIR.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            print(f"    {p.name:<24} {p.stat().st_size / 1024:>8.0f} KB")

    footer("测试图片下载完成")


if __name__ == "__main__":
    main()
