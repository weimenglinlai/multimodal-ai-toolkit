# -*- coding: utf-8 -*-
"""
基于 curl 的模型文件下载器（绕过 hf-mirror 的 xet 超时）
=========================================================
实测问题：用 huggingface_hub 的 snapshot_download 走 hf-mirror 时，
xet 协议的大文件（>500MB）反复 "The read operation timed out"，
下载到一半就断，重试也断，效率极低。

解决：直接用 curl 打 hf-mirror 的 resolve 端点，
     支持 -C - 断点续传，多文件并行，逐个校验 Content-Length。

用法：
    python scripts/curl_download.py <repo_id> <file1> [file2 ...] --dest <目录>
    python scripts/curl_download.py openai/clip-vit-base-patch32 pytorch_model.bin --dest model/clip-vit-base-patch32
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def remote_size(url: str) -> int:
    """用 HEAD 请求拿 Content-Length"""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def download(repo: str, rel_path: str, dest: Path, retries: int = 6) -> bool:
    url = f"{MIRROR}/{repo}/resolve/main/{rel_path}"
    out = dest / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        expected = remote_size(url)
        if expected and out.stat().st_size == expected:
            print(f"      [已存在] {rel_path}  ({out.stat().st_size / 1e6:.0f} MB)")
            return True
        print(f"      [续传] {rel_path} 已有 {out.stat().st_size / 1e6:.0f} MB，继续下载")

    expected = remote_size(url)
    for attempt in range(1, retries + 1):
        cmd = ["curl", "-L", "-C", "-", "--fail", "--retry", "3",
               "--connect-timeout", "20", "--max-time", "1800",
               "-o", str(out), url]
        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", "ignore")[-200:]
            print(f"      [重试 {attempt}/{retries}] {rel_path} 中断: {err.strip()[:120]}")
            if out.exists() and expected and out.stat().st_size >= expected:
                break
            time.sleep(3)
            continue

        got = out.stat().st_size if out.exists() else 0
        if expected and got < expected:
            print(f"      [重试 {attempt}/{retries}] {rel_path} 不完整 "
                  f"{got / 1e6:.0f}/{expected / 1e6:.0f} MB")
            time.sleep(3)
            continue
        print(f"      [OK] {rel_path}  {got / 1e6:.0f} MB  用时 {time.time() - t0:.0f}s")
        return True

    print(f"      [失败] {rel_path} 多次重试仍未完成")
    return False


def main():
    ap = argparse.ArgumentParser(description="curl 版模型下载器")
    ap.add_argument("repo", help="仓库 id，如 openai/clip-vit-base-patch32")
    ap.add_argument("files", nargs="+", help="仓库内相对路径")
    ap.add_argument("--dest", required=True, help="本地目标目录")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"仓库：{args.repo}")
    print(f"目标：{dest}")
    print(f"镜像：{MIRROR}")
    print()

    ok = True
    for f in args.files:
        if not download(args.repo, f, dest):
            ok = False
    print()
    print("全部完成" if ok else "存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
