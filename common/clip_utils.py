# -*- coding: utf-8 -*-
"""
CLIP 特征提取与相似度计算工具
==============================
⚠️ 两个实测得到的关键事实（本项目 transformers 5.12.1 环境）：

1. `model(**inputs).logits_per_image` 不是余弦相似度，而是
       logits = logit_scale × cosine      （logit_scale ≈ 100）
   所以真实余弦相似度必须用 `logits / logit_scale` 还原。
   参考文档示例代码里直接打印 logits 属于「缩放后的对数概率」，
   数值在 20~35 之间，不是 0~1 的相似度，报告里不能当相似度写。

2. transformers 5.x 中 `get_text_features()` / `get_image_features()`
   返回的是 `BaseModelOutputWithPooling` 而不是 Tensor，
   且其中的特征向量**未做 L2 归一化**（实测模长 31.25 / 12.09）。
   必须自己取 pooler_output 并手动归一化，才能算点积余弦。

本模块把这些坑统一封住，任务一、任务二拓展1 都用它。
"""
import torch
import torch.nn.functional as F


def _to_tensor(out):
    """把 transformers 5.x 的 Output 对象取成 Tensor；兼容旧版直接返回 Tensor"""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if isinstance(out, (tuple, list)) and len(out) and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"无法从 {type(out).__name__} 提取张量")


def get_text_features(model, input_ids, attention_mask=None):
    """取文本特征（未归一化 Tensor）"""
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return _to_tensor(model.get_text_features(**kwargs))


def get_image_features(model, pixel_values):
    """取图像特征（未归一化 Tensor）"""
    return _to_tensor(model.get_image_features(pixel_values=pixel_values))


def cosine_matrix(text_feats, image_feats):
    """计算文本×图像的余弦相似度矩阵（行=文本，列=图像）"""
    t = F.normalize(text_feats.float(), dim=-1)
    i = F.normalize(image_feats.float(), dim=-1)
    return t @ i.T


def logits_to_cosine(logits, model):
    """把 model 输出的 logits 还原为余弦相似度"""
    scale = model.logit_scale.exp().item()
    return logits / scale


def clip_similarity(model, processor, texts, images, device):
    """
    一步算出 文本×图像 的余弦相似度矩阵（float32, CPU）。

    返回 (sim_matrix, logits)
      sim_matrix: shape (len(texts), len(images))，值为 -1~1 的余弦相似度
      logits    : shape (len(images), len(texts))，模型原始输出
                  = sim_matrix.T × logit_scale
    注意两个矩阵方向不同：sim 是「行=文本」，logits 是「行=图像」，
    这是 transformers 的 logits_per_image 约定，容易搞错。
    """
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        tfeat = get_text_features(
            model,
            inputs["input_ids"],
            inputs.get("attention_mask"),
        )
        ifeat = get_image_features(model, inputs["pixel_values"])

    sim = cosine_matrix(tfeat, ifeat).cpu()          # (M, N)
    return sim, out.logits_per_image.cpu()           # (N, M)


def softmax_over_texts(logits):
    """
    每张图片在「所有文本候选」上的分布（图像→文本方向）。
    输入 logits 形状 (N, M) = (图片数, 文本数)，沿文本维(dim=1)归一化。
    返回 (N, M)。
    """
    return torch.softmax(logits, dim=1)


def softmax_over_images(sim, text_index):
    """
    某条文本在「所有图片候选」上的分布（文本→图像方向）。
    输入 sim 形状 (M, N) = (文本数, 图片数)，取第 text_index 行沿图片维归一化。
    返回 (N,)。
    """
    return torch.softmax(sim[text_index], dim=0)
