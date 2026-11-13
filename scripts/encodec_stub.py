# -*- coding: utf-8 -*-
"""
encodec 占位模块（Windows / Python 3.13 兼容补丁）
=================================================
为什么需要这个 stub：

  vocos 的 feature_extractors.py 顶层有 `from encodec import EncodecModel`，
  而 encodec 官方只发布到 cp311 的预编译 wheel，Python 3.13 下装不了，
  pip 会尝试从源码编译 Rust 扩展并最终失败。

  但 ChatTTS 实际走的是它自己的 Mel 特征提取器，
  vocos 里的 EncodecFeatures 只用于 isinstance 类型判断，永远不会被实例化。

  所以这里提供一个最小 stub：只要模块能 import、类能存在即可，
  真正用到 EncodecModel 时会明确报错，而 ChatTTS 路径永远不会走到那里。

本文件由 scripts/setup_chattts_env.bat 自动部署到 venv_chattts 的 site-packages。
"""

__version__ = "0.0.0-stub"


class EncodecModel:  # pragma: no cover - 占位，仅为通过 vocos 的 import
    """占位实现。ChatTTS 使用 Mel 特征，不会实例化本类。"""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "encodec 是占位模块：当前环境（Python 3.13）无对应预编译 wheel。"
            "ChatTTS 走 Mel 特征提取路径，不需要真实 encodec，"
            "如果你确实需要 encodec，请改用 Python 3.11 环境安装官方包。"
        )


def get_model(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("encodec 是占位模块，不提供 get_model。")


__all__ = ["EncodecModel", "get_model"]
