# -*- coding: utf-8 -*-
"""
自动补齐 ChatTTS 缺失依赖
=========================
ChatTTS 0.2.5 的依赖链比声明的长（vocos→encodec、vector_quantize_pytorch→einx→frozendict/loguru …），
逐个 pip install 很慢。本脚本做「导入报错 → 装包 → 再导入」的自动循环。

用法：
    venv_chattts\\Scripts\\python.exe scripts\\fix_chattts_deps.py
    venv_chattts\\Scripts\\python.exe scripts\\fix_chattts_deps.py --max 30
"""
import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path

# 导入名 → pip 包名（不一致的才需要映射）
NAME_MAP = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "IPython": "ipython",
}

# 已经用 stub 解决的（不是真包）
STUBBED = {"encodec"}


def pip_install(pkg: str) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps",
           "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
           "--trusted-host", "pypi.tuna.tsinghua.edu.cn", pkg]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            print(f"      安装 {pkg} 成功")
            return True
        print(f"      安装 {pkg} 失败: {r.stderr.strip()[-200:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"      安装 {pkg} 超时")
        return False


def find_missing():
    """跑一遍导入，抓第一个 ModuleNotFoundError"""
    code = (
        "import ChatTTS\n"
        "from ChatTTS.core import Chat\n"
        "print('IMPORT_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=300)
    if "IMPORT_OK" in r.stdout:
        return None, True
    m = re.search(r"No module named '([^']+)'", r.stderr)
    if m:
        return m.group(1), False
    # 其它报错（非缺包）
    return None, False


def main():
    ap = argparse.ArgumentParser(description="自动补齐 ChatTTS 依赖")
    ap.add_argument("--max", type=int, default=30, help="最多循环次数")
    args = ap.parse_args()

    print("开始检查 ChatTTS 导入链…")
    for i in range(1, args.max + 1):
        missing, ok = find_missing()
        if ok:
            print(f"\n第 {i} 次检查：导入成功，全部依赖已就位。")
            return
        if missing is None:
            print(f"\n第 {i} 次检查：出现非缺包错误：")
            r = subprocess.run([sys.executable, "-c",
                                "import ChatTTS\nfrom ChatTTS.core import Chat"],
                               capture_output=True, text=True)
            print(r.stderr.strip()[-1500:])
            return

        pkg = NAME_MAP.get(missing, missing)
        if missing in STUBBED:
            print(f"  缺少 {missing}（已用 stub 处理），跳过")
            return

        print(f"第 {i} 次检查：缺少模块 {missing} → 安装 {pkg}")
        if not pip_install(pkg):
            print("  安装失败，停止。")
            return

    print("\n达到最大循环次数仍未成功，请手动检查。")


if __name__ == "__main__":
    main()
