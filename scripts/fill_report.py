# -*- coding: utf-8 -*-
"""
填充《AI》任务3 报告（第四、五、六节）
==========================================================
用法：
    python scripts/fill_report.py <目标docx> [--dry]

设计要点（为什么不走 editor_sdk 的 doc_insert_markdown）：
    该文档主体是一张 9 行大表格，要往第 5/6/7 行的单元格里整段写入
    几十个段落和十几张图片。用坐标式 MCP 接口需要反复 resolve 结构、
    坐标随每次写入漂移，极易错位；python-docx 直接定位到 cell 再顺序
    append，稳定得多，也不会触发 editor_sdk 的 markdown 导入复制 bug。

格式（按模板要求）：中文宋体、英文 Times New Roman、小四（12pt）、
单倍行距、段前段后 0 行、正文首行缩进 2 字符。
"""
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
SD_OUT = ROOT / "sd" / "outputs"

# 报告要引用的实测数字（全部来自 logs/ 与 结果汇总.md）
CLIP_1N = "0.4970"
CLIP_1N_GAP = "0.1801"
SD_TIMES = ("2.65", "1.96", "1.96")
SD_STEPS = [
    ("10", "1.08", "108.0", "1.00"),
    ("20", "1.91", "95.7", "1.77"),
    ("30", "2.75", "91.5", "2.54"),
    ("50", "4.41", "88.1", "4.08"),
    ("100", "8.60", "86.0", "7.97"),
]


# ---------------------------------------------------------------- 格式工具
def style_run(run, size=12, bold=False, ascii_font="Times New Roman",
              ea_font="宋体"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = ascii_font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), ascii_font)
    rfonts.set(qn("w:hAnsi"), ascii_font)
    rfonts.set(qn("w:eastAsia"), ea_font)
    return run


def fmt_para(p, indent_chars=2, align=None, before=0, after=0, spacing=1.0):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    if indent_chars:
        pf.first_line_indent = Pt(12 * indent_chars)
    else:
        pf.first_line_indent = Pt(0)
    if align is not None:
        p.alignment = align
    return p


def add_text(cell, text, indent=2, bold=False, size=12, align=None,
             before=0, after=0):
    p = cell.add_paragraph()
    fmt_para(p, indent_chars=indent, align=align, before=before, after=after)
    style_run(p.add_run(text), size=size, bold=bold)
    return p


def add_caption(cell, text):
    """图注：居中、小四、无缩进"""
    p = cell.add_paragraph()
    fmt_para(p, indent_chars=0, align=WD_ALIGN_PARAGRAPH.CENTER, before=6,
             after=6)
    style_run(p.add_run(text), size=12)
    return p


def add_note(cell, text):
    """待插入截图的提示行：加粗方括号，便于用户查找"""
    p = cell.add_paragraph()
    fmt_para(p, indent_chars=0, before=3, after=3)
    style_run(p.add_run(text), size=12, bold=True)
    return p


def add_image(cell, img_path, width_cm=9.0):
    p = cell.add_paragraph()
    fmt_para(p, indent_chars=0, align=WD_ALIGN_PARAGRAPH.CENTER, before=4,
             after=4)
    p.add_run().add_picture(str(img_path), width=Cm(width_cm))
    return p


def clear_cell(cell, keep_first=True):
    """清空单元格内全部段落，只留一个空段

    注意：留下的这个空段要调用方在写完内容后用 drop_leading_empty() 删掉，
    否则正文前会多出一行空白。
    """
    for p in list(cell.paragraphs)[1 if keep_first else 0:]:
        p._element.getparent().remove(p._element)
    if not cell.paragraphs:
        cell.add_paragraph()
    # 清掉首段的文字
    first = cell.paragraphs[0]
    for r in list(first.runs):
        r._element.getparent().remove(r._element)


def drop_leading_empty(cell):
    """删掉开头的空段落（clear_cell 留下的）"""
    ps = cell.paragraphs
    if len(ps) > 1 and not ps[0].text.strip() and not ps[0].runs:
        ps[0]._element.getparent().remove(ps[0]._element)


# ---------------------------------------------------------------- 第四节
def fill_section4(cell):
    clear_cell(cell)
    add_text(cell, "四、实践操作方法与步骤（原始数据记录可另附）",
             indent=0, bold=True)

    add_text(cell,
             "本次实践在 64 位 Windows 操作系统上完成。编程语言为 Python 3.13.4，"
             "深度学习框架为 PyTorch 2.6.0+cu124，图像生成库为 diffusers 0.42.1，"
             "模型接口库为 transformers 5.12.1，推理设备为 NVIDIA GeForce RTX 4060 "
             "Laptop GPU（8GB 显存）。所有模型权重预先下载至本地 model 目录，"
             "实践过程不依赖外网，保证结果可复现。")

    # ------------------------------ （一）CLIP
    add_text(cell, "（一）基于CLIP的文本-图像匹配", indent=0)
    add_text(cell,
             "选用中文 CLIP 模型 chinese-clip-vit-base-patch16，本地权重约 718MB。"
             "测试图片共 6 张，内容分别为蜜蜂采蜜、爱尔兰酒吧、老爷车、虎斑猫、"
             "农田航拍与城市夜景，覆盖自然、建筑、交通等不同类别。")
    add_text(cell,
             "实现时统一封装了特征提取流程：图像经处理器缩放并归一化后送入视觉编码器，"
             "文本经分词器编码后送入文本编码器，两侧均取池化输出并做 L2 归一化，"
             "再以归一化向量的内积作为余弦相似度。这样处理的原因是 transformers 5.x 的"
             "特征接口返回的是含多层输出的对象，且特征本身未归一化，直接点积得到的不是余弦值。")

    add_text(cell, "1．1条中文语句与N张图片的匹配", indent=0)
    add_text(cell, "实现步骤：")
    add_text(cell, "1. 加载分词器与模型，将模型移至 GPU 并切换为评估模式；", indent=0)
    add_text(cell, "2. 读取待测图片，经处理器完成缩放与归一化，转为像素张量；", indent=0)
    add_text(cell, "3. 将候选中文语句批量分词，得到文本张量；", indent=0)
    add_text(cell, "4. 分别提取图像特征与文本特征，取池化输出后做 L2 归一化；", indent=0)
    add_text(cell, "5. 以内积计算余弦相似度，按降序排序后取首位作为匹配结果。", indent=0)
    add_note(cell,
             "［待插入截图1］代码：clip_cn/task1_1N_match.py 全文（约 1—120 行），"
             "截图需包含窗口上方打印的、、院系、实践室信息。")
    add_note(cell,
             "［待插入截图2］运行结果：以「一只蜜蜂停在粉色的花上采蜜」为查询语句运行上述脚本，"
             "截图需包含 6 张候选图的相似度排序表与最匹配结论。")

    add_text(cell, "2．M条中文语句与N张图片的匹配", indent=0)
    add_text(cell, "实现步骤：")
    add_text(cell,
             "1. 一次性编码全部 N 张图片，得到图像特征矩阵；", indent=0)
    add_text(cell,
             "2. 一次性编码全部 M 条语句，得到文本特征矩阵；", indent=0)
    add_text(cell,
             "3. 计算两个矩阵的余弦相似度矩阵，形状为 M×N；", indent=0)
    add_text(cell,
             "4. 沿图像维度做 softmax 归一化，得到每条语句在 N 张图上的概率分布；", indent=0)
    add_text(cell,
             "5. 取概率最大的列作为匹配结果，并统计 Top-1、Top-2 命中率。", indent=0)
    add_note(cell,
             "［待插入截图3］代码：clip_cn/task2_MN_match.py 全文，需包含个人信息输出。")
    add_note(cell,
             "［待插入截图4］运行结果：M=6、N=6 的相似度矩阵热力表与 Top-1／Top-2 命中率统计。")

    add_text(cell, "3．拓展任务1：其他CLIP模型实践", indent=0)
    add_text(cell,
             "为验证模型语言能力对检索效果的影响，引入原版 CLIP 模型 "
             "clip-vit-base-patch32 作为对照，在同一组图片上分别使用中文描述与"
             "英文描述进行检索，并与中文 CLIP 的结果比较。")
    add_text(cell, "实现步骤：")
    add_text(cell,
             "1. 分别加载原版 CLIP 与中文 CLIP 两套权重，统一放到 GPU 上；", indent=0)
    add_text(cell,
             "2. 为每张图片准备中文、英文两种描述，保证语义一致；", indent=0)
    add_text(cell,
             "3. 按模型与描述语言的四种组合分别计算 Top-1 并统计命中率；", indent=0)
    add_text(cell,
             "4. 记录同一张图片在不同模型下的相似度数值，用于横向比较。", indent=0)
    add_note(cell,
             "［待插入截图5］代码：clip_cn/task3_other_clip_compare.py 全文，需包含个人信息输出。")
    add_note(cell,
             "［待插入截图6］运行结果：四种组合的 Top-1 命中率统计，"
             "以及同一图片在中文 CLIP 与原版 CLIP 下的相似度对照。")
    add_text(cell,
             "需要说明的是，拓展任务2（ModelScope 平台中文 CLIP 在线实践）依托平台在线环境运行，"
             "其输入输出由平台托管，无法在本地代码中复现个人信息输出，本次未纳入代码工程。")

    # ------------------------------ （二）Stable Diffusion
    add_text(cell, "（二）基于Stable Diffusion的文本-图像生成", indent=0)
    add_text(cell,
             "选用轻量级扩散模型 tiny-sd，输出分辨率为 512×512。为保证变量可控，"
             "固定随机种子 seed=42、提示词引导系数 CFG=7.5，"
             "模型以 float16 精度在 GPU 上推理，仅改变提示词或推理步数两个变量。")

    add_text(cell, "1．子任务1：提示词工程探索——关键词对生成结果的影响", indent=0)
    add_text(cell, "实现步骤：")
    add_text(cell,
             "1. 加载文本编码器、U-Net 与 VAE 三个组件，组装成完整推理管线；", indent=0)
    add_text(cell,
             "2. 固定 seed 与推理步数（30 步），准备三组语义递进的提示词；", indent=0)
    add_text(cell,
             "3. 三组提示词分别为：仅主体「A cat」、补充细节「A white cat with blue eyes, "
             "sitting on a wooden table, watercolor style」、以及细节版调换语序"
             "「Watercolor style, a wooden table with a white cat sitting on it, blue eyes」；",
             indent=0)
    add_text(cell,
             "4. 逐组推理并记录生成耗时，保存结果图片用于对比。", indent=0)
    add_note(cell,
             "［待插入截图7］代码：sd/task1_prompt_engineering.py 全文，需包含个人信息输出。")
    add_note(cell,
             "［待插入截图8］运行结果：三组提示词的终端输出，含个人信息与各组耗时统计。")

    add_text(cell, "生成结果如下三图所示，三组共用同一随机种子，差异只可能来自提示词本身。")
    for fn, cap in [
        ("task1_base.png", "图4-1  基础版提示词生成结果（A cat）"),
        ("task1_detail.png", "图4-2  细节版提示词生成结果"),
        ("task1_reorder.png", "图4-3  语序调整版提示词生成结果"),
    ]:
        p = SD_OUT / fn
        if p.exists():
            add_image(cell, p, width_cm=8.0)
            add_caption(cell, cap)

    add_text(cell, "2．子任务2：推理步数的权衡——速度与质量的平衡", indent=0)
    add_text(cell, "实现步骤：")
    add_text(cell,
             "1. 固定提示词与随机种子，仅将推理步数依次设为 10、20、30、50、100；", indent=0)
    add_text(cell,
             "2. 每次推理前先执行一次 2 步的预热，排除 CUDA 初始化带来的额外开销；", indent=0)
    add_text(cell,
             "3. 用计时器记录每组的总耗时，除以步数得到单步耗时；", indent=0)
    add_text(cell,
             "4. 保存每组结果图片，横向比较画质随步数的变化。", indent=0)
    add_note(cell,
             "［待插入截图9］代码：sd/task2_inference_steps.py 全文，需包含个人信息输出。")
    add_note(cell,
             "［待插入截图10］运行结果：五组步数对应的总耗时、单步耗时与相对耗时统计表。")
    add_text(cell, "各组耗时实测数据如下：")
    for steps, total, per, ratio in SD_STEPS:
        add_text(cell,
                 f"num_inference_steps = {steps}：总耗时 {total} 秒，"
                 f"单步约 {per} 毫秒，为 10 步耗时的 {ratio} 倍。", indent=0)

    add_text(cell, "生成结果如下五图所示，提示词固定为「a futuristic city at night, "
                   "neon lights, cyberpunk style」。")
    for steps, cap in [
        ("010", "图4-4  推理步数 10 的生成结果"),
        ("020", "图4-5  推理步数 20 的生成结果"),
        ("030", "图4-6  推理步数 30 的生成结果"),
        ("050", "图4-7  推理步数 50 的生成结果"),
        ("100", "图4-8  推理步数 100 的生成结果"),
    ]:
        p = SD_OUT / f"task2_steps{steps}.png"
        if p.exists():
            add_image(cell, p, width_cm=7.5)
            add_caption(cell, cap)

    add_text(cell, "3．拓展任务1：自动评估生成质量——用CLIP计算提示词与图像的相似度", indent=0)
    add_text(cell, "实现步骤：")
    add_text(cell,
             "1. 复用任务一的中文 CLIP 特征提取流程，把生成图片与原文提示词分别编码；", indent=0)
    add_text(cell,
             "2. 计算两者的余弦相似度，作为图文语义对齐程度的量化指标；", indent=0)
    add_text(cell,
             "3. 分别从提示词维度（4 组不同提示词）与推理步数维度（5 组不同步数）"
             "做两组对照评估；", indent=0)
    add_text(cell,
             "4. 统计相似度随变量的变化趋势，判断边际收益拐点。", indent=0)
    add_note(cell,
             "［待插入截图11］代码：sd/task3_clip_eval.py 全文，需包含个人信息输出。")
    add_note(cell,
             "［待插入截图12］运行结果：实践A（不同提示词）与实践B（不同步数）的 CLIP 相似度对比表。")
    add_text(cell,
             "拓展任务2（ModelScope 平台中文 Stable Diffusion 在线实践）同样依托平台在线环境，"
             "输入输出由平台托管，无法在本地代码中输出个人信息，本次未纳入代码工程。")

    # ------------------------------ （三）Whisper + ChatTTS
    add_text(cell, "（三）Whisper语音转录+ChatTTS语音合成", indent=0)

    add_text(cell, "1．Whisper实践任务", indent=0)
    add_text(cell,
             "Whisper 要求输入 16kHz 单声道音频。实测发现即使输入已是 16kHz 的 wav 文件，"
             "其音频读取模块仍会调用 ffmpeg 做解码，因此需预先配置 ffmpeg 可执行文件。"
             "本实践将 ffmpeg 置于项目 bin 目录并在程序启动时加入 PATH，不改动系统环境变量。")
    add_text(cell, "实现步骤：")
    add_text(cell,
             "1. 使用 16kHz、单声道、时长约 10.2 秒的测试音频，内容为自报"
             "与一句日常用语；", indent=0)
    add_text(cell,
             "2. 加载 base 规模的 Whisper 模型，以中文为指定语言执行转录；", indent=0)
    add_text(cell,
             "3. 记录推理耗时，并按实时率 RTF＝推理耗时／音频时长衡量效率；", indent=0)
    add_text(cell,
             "4. 将识别文本与标准答案比对，统计、的识别正确率，"
             "用最长公共子串长度量化的偏差程度；", indent=0)
    add_text(cell,
             "5. 追加 transformers 实现版本作为对照，两者加载同一份权重，"
             "用于验证不同推理框架的输出差异。", indent=0)
    add_note(cell,
             "［待插入截图13］代码：whisper/task1_realtime_record.py 与 whisper/whisper_test.py，"
             "截图需包含个人信息输出。")
    add_note(cell,
             "［待插入截图14］实时识别结果：麦克风录音时说出与，"
             "截图需同时包含 Whisper 识别出的、以及程序打印的个人信息。")
    add_note(cell,
             "［待插入截图15］拓展任务：whisper/task2_hf_whisper.py 的代码与运行结果，"
             "需包含识别文本、数字片段归一化输出与个人信息。")

    add_text(cell, "2．ChatTTS实践任务", indent=0)
    add_text(cell,
             "ChatTTS 0.2.5 的接口与常见示例存在差异：模型类名为 Chat，权重需从本地目录加载"
             "（其内置的在线下载通道依赖境外资源，国内网络无法直连），"
             "推理参数需通过 InferCodeParams 传入，且返回的波形为 float32 浮点数组，"
             "写入 wav 文件前必须量化成 16 位整型，否则会出现削波失真。"
             "由于该库依赖 transformers 4.x，与主环境的 5.x 冲突，"
             "因此单独建立了名为 venv_chattts 的虚拟环境。")
    add_text(cell, "实现步骤：")
    add_text(cell,
             "1. 建立独立虚拟环境并安装 ChatTTS 及其依赖，权重下载至本地 model 目录；", indent=0)
    add_text(cell,
             "2. 从本地目录加载模型，设备选择 GPU，采样率 24000Hz；", indent=0)
    add_text(cell,
             "3. 随机采样一个音色嵌入作为说话人特征，保证三组文本音色一致；", indent=0)
    add_text(cell,
             "4. 准备日常对话、新闻播报、情感表达三组测试文本，逐条合成；", indent=0)
    add_text(cell,
             "5. 将浮点波形量化为 int16 后写入 wav 文件，记录合成耗时与音频时长；", indent=0)
    add_text(cell,
             "6. 在 minimax 在线平台使用「新闻女声」音色、无水印 WAV 格式合成同一组文本，"
             "作为对照；", indent=0)
    add_text(cell,
             "7. 把两个平台的音频统一送入 Whisper 做反向识别，"
             "以字符错误率 CER 作为发音准确度的客观指标。", indent=0)
    add_note(cell,
             "［待插入截图16］ChatTTS 合成：chattts/chat_tts_synth.py 的代码与运行输出，"
             "需包含三组文本的合成耗时、音频时长与个人信息，"
             "并附音频文件列表，参照参考文档图3.38。")
    add_note(cell,
             "［待插入截图17］minimax 官网合成：打开 minimax 语音合成页面，"
             "截取输入三组文本及生成历史的界面。")
    add_note(cell,
             "［待插入截图18］客观指标评估：scripts/eval_tts_quality.py 的运行结果，"
             "包含两平台 CER 对比表与个人信息。")
    drop_leading_empty(cell)


# ---------------------------------------------------------------- 第五节
def fill_section5(cell):
    clear_cell(cell)
    add_text(cell, "五、实践结果与分析", indent=0, bold=True)

    add_text(cell, "（一）基于CLIP的文本-图像匹配结果分析", indent=0)
    add_text(cell,
             f"1条语句对6张图片的检索中，查询语句「一只蜜蜂停在粉色的花上采蜜」的"
             f"正确匹配项 bee_on_flower.jpg 余弦相似度为 {CLIP_1N}，"
             f"以 {CLIP_1N_GAP} 的明显优势领先第二名。其余五个候选的相似度集中在 "
             f"0.26～0.32 的窄区间内，说明模型对「与该查询无关」的图像给出了"
             f"彼此接近的低分，区分度主要体现在正确项上，语义对齐是有效的。")
    add_text(cell,
             "M条语句对N张图片的检索中，6×6 相似度矩阵的对角线全部为每行最大值，"
             "文本到图像的 Recall@1 与 Recall@2 均达到 100%。"
             "这说明在图文配对明确、类别差异较大的小型图库上，"
             "中文 CLIP 的跨模态对齐已经足够可靠，检索不存在混淆。")
    add_text(cell,
             "模型对照实践显示出显著差异：同一张蜜蜂图片，中文 CLIP 对中文描述给出 "
             "0.4992 的相似度，而原版 CLIP 对语义一致的英文描述只有 0.3068，"
             "对中文描述更低。命中率统计也印证了这一点——原版 CLIP 配中文描述仅 66.7%，"
             "换成英文描述后升至 100%，中文 CLIP 配中文描述同样为 100%。"
             "其原因在于原版 CLIP 的文本编码器主要在英文语料上训练，"
             "中文词的表示未落在与图像对齐的语义空间中；"
             "这也从实践角度说明了开展中文多模态预训练的必要性。")

    add_text(cell, "（二）基于Stable Diffusion的文本-图像生成结果分析", indent=0)
    add_text(cell,
             f"提示词工程实践表明，关键词的丰富程度直接决定生成内容的可控性。"
             f"基础版「A cat」只约束了主体，生成结果是一只虎斑猫的实景照片，"
             f"毛色、眼睛颜色、场景与画风均未受控；细节版补充了毛色、眼色、姿态、"
             f"桌面与画风五个约束后，白猫、蓝眼、木桌与水彩质感全部得到体现；"
             f"语序版与细节版语义完全相同，仅把画风词提到句首，"
             f"生成结果的水彩晕染效果明显更强烈、整幅画面风格更统一。"
             f"三组耗时分别为 {SD_TIMES[0]}、{SD_TIMES[1]}、{SD_TIMES[2]} 秒，"
             f"差异在测量噪声范围内，说明画面差异确实只来自提示词本身。"
             f"可以认为文本编码器对处于句首的修饰词赋予了更高的权重。")
    add_text(cell,
             "推理步数实践显示出清晰的线性关系：总耗时由 10 步的 1.08 秒增至 100 步的 "
             "8.60 秒，相对倍数 7.97，与步数倍数基本一致，说明每步计算量恒定，"
             "单步耗时从 108 毫秒缓慢下降到 86 毫秒，是 GPU 由冷启动转为稳定负载后的正常表现。"
             "而从画质看，20 至 30 步时建筑结构与光影关系已基本成型，"
             "继续增加到 50、100 步，细节纹理的提升幅度远小于耗时增幅，边际收益明显递减。"
             "综合考虑，实际应用中把步数取在 20 至 30 之间即可满足绝大多数需求。")
    add_text(cell,
             "CLIP 相似度评估给出了与主观观察一致的量化结果。提示词维度上，"
             "相似度由「A cat」的 0.2745 依次升至复杂描述 0.3752、细节版 0.4009、"
             "语序版 0.4372，说明描述越具体、越贴合生成结果，图文对齐程度越高。"
             "步数维度上，相似度仅由 0.3432 微升至 0.3521，且在 30 步之后基本持平，"
             "与前述画质观察相互印证。需要注意的是，CLIP 相似度只衡量语义对齐，"
             "不衡量构图美感与画面伪影，因此它适合作为辅助指标，不能完全替代人工判断。")

    add_text(cell, "（三）Whisper语音转录与ChatTTS语音合成结果分析", indent=0)
    add_text(cell,
             "Whisper base 模型对 10.2 秒音频的转录耗时为 0.77 秒，"
             "「」完整识别正确，「」的最长公共子串长度为 "
             "10/12，偏差集中在个别数字上。这符合 Whisper 对长数字串的一贯表现——"
             "数字并非独立词元，连续多位时容易出现漏位或串位。"
             "顺带的分块实践也印证了这一点：transformers 实现按 30 秒定长窗口切分，"
             "把完整切成了「20230 1170 150」三段，字符全部正确但被插入了空格；"
             "该版本推理耗时 1.38 秒，实时率 0.14，仍远快于实时。"
             "可见两种实现的差异主要来自分块策略，而非模型能力。")
    add_text(cell,
             "ChatTTS 与 minimax 的对比呈现出较明显的差距。以 Whisper 反向识别后的"
             "字符错误率衡量，ChatTTS 三组分别为 0.00%、25.00%、12.50%，平均 12.50%；"
             "minimax 分别为 0.00%、3.57%、0.00%，平均 1.19%。"
             "差距主要集中在长句上：新闻播报一组中，ChatTTS 把「本市」读成「本事」、"
             "「众多企业」读成「中国体验」，属于同音字混淆；minimax 仅把「据最新报道」"
             "识别为「据说新报导」，偏差明显更小。短句上两者持平。")
    add_text(cell,
             "客观指标还揭示出一处容易被误判为音质差异的因素：三组文本中 minimax 的"
             "响度均值为 −18.0、−18.4、−19.3 dB，波动很小；ChatTTS 为 −25.5、−20.3、"
             "−23.0 dB，不仅整体偏低且波动较大。这说明 minimax 在输出端做了响度归一化，"
             "而 ChatTTS 保留了原始动态范围。听感上「minimax 更响、更清楚」的一部分原因"
             "正在于此，并非纯粹的音质优劣。时长方面 minimax 三组分别比 ChatTTS 长 "
             "0.97、0.92、0.35 秒，语速更慢、停顿更充分，这也是长句准确率更高的原因之一。"
             "需要指出的是，字符错误率只能反映「说得清不清楚」，反映不了韵律与情感，"
             "情感表达一组的实际听感仍需人工判断，这也是客观指标无法替代主观评价的地方。")
    drop_leading_empty(cell)


# ---------------------------------------------------------------- 第六节
def fill_section6(cell):
    clear_cell(cell)
    add_text(cell, "六、实践体会", indent=0, bold=True)

    add_text(cell,
             "本次实践围绕 CLIP、Stable Diffusion 与 Whisper、ChatTTS 三组模型展开，"
             "覆盖了图文检索、文生图与语音识别合成三类典型的多模态任务。"
             "通过完整走通模型下载、环境配置、推理调用与结果评估的流程，"
             "对多模态大模型的工作机理与工程落地有了较为具体的认识。")
    add_text(cell,
             "第一点体会是，跨模态对齐的效果高度依赖训练语料的语言分布。"
             "同样一张图片、同样语义的描述，中文 CLIP 的相似度接近 0.5，"
             "而原版 CLIP 面对中文描述时命中率只有 66.7%，换成英文即恢复至 100%。"
             "这组对照直观地说明，多模态模型的能力边界并不只由网络结构决定，"
             "预训练阶段的语种覆盖同样关键。此前对「中文化」的理解停留在概念层面，"
             "经过这次实测才有了量化依据。")
    add_text(cell,
             "第二点体会是，工程细节对结果的影响常常被低估。"
             "ChatTTS 的波形是浮点数组，若不量化为整型直接写入文件，"
             "得到的音频会严重削波，下游识别结果全为空，而现象上看不出任何报错；"
             "Whisper 即便读到 16kHz 的标准音频，仍会调用外部解码器，"
             "缺少该组件时程序直接中断；特征提取接口在版本升级后返回类型与归一化状态发生变化，"
             "若不核对就会把缩放后的对数值误当作相似度。"
             "这些问题都不属于算法层面，却足以让实践结论完全失真。"
             "因此在建模之外，养成对中间结果做数值校验的习惯十分必要，"
             "例如检查音频的峰值与有效值、检查相似度的取值范围是否合理。")
    add_text(cell,
             "第三点体会是，评价指标各有适用边界，需要组合使用。"
             "CLIP 相似度能够量化图文语义对齐程度，却不衡量构图美感与画面伪影；"
             "推理步数增加到一定程度后相似度趋于饱和，单看指标容易忽略耗时的急剧上升；"
             "字符错误率可以反映发音清晰度，却无法评价韵律与情感表达。"
             "本次实践把客观指标与人工观察结合起来判断，结论才比较可靠。"
             "这一经验对今后设计模型评估方案具有直接的参考价值。")
    add_text(cell,
             "此外，在复现过程中也体会到环境隔离的重要性。"
             "ChatTTS 依赖的接口库版本与主环境冲突，若不单独建立虚拟环境，"
             "将导致主环境其他实践无法运行。"
             "通过为冲突组件建立独立环境、并把模型权重统一落在本地目录，"
             "既保证了实践的可复现性，也避免了对全局环境造成污染。"
             "总体而言，本次实践不仅加深了对多模态模型原理的理解，"
             "也提升了对实践流程规范性、结果可验证性的重视程度。")
    drop_leading_empty(cell)


def tidy_step_indent(cell):
    """把「1. 2. 3.」这类编号步骤的首行缩进调成与正文一致

    写入时这些行用的是 indent=0（贴左边距），视觉上会跟章节标题同级、
    层级不清；统一改成 2 字符缩进，和「实现步骤：」下面的正文对齐。
    """
    import re
    for p in cell.paragraphs:
        if re.match(r"^\d+\.\s", p.text.strip()):
            p.paragraph_format.first_line_indent = Pt(24)


def main():
    if len(sys.argv) < 2:
        print("用法: python fill_report.py <目标docx>")
        return 1
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"文件不存在: {target}")
        return 1

    d = docx.Document(str(target))
    t = d.tables[0]
    if len(t.rows) < 8:
        print(f"表格行数异常: {len(t.rows)}")
        return 1

    fill_section4(t.rows[5].cells[0])
    fill_section5(t.rows[6].cells[0])
    fill_section6(t.rows[7].cells[0])

    for ri in (5, 6, 7):
        tidy_step_indent(t.rows[ri].cells[0])

    d.save(str(target))
    print(f"已写入: {target}")
    print(f"  第四节段落数: {len(t.rows[5].cells[0].paragraphs)}")
    print(f"  第五节段落数: {len(t.rows[6].cells[0].paragraphs)}")
    print(f"  第六节段落数: {len(t.rows[7].cells[0].paragraphs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
