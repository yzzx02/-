"""
lipid_classifier.py
----------------------------------------------------------
用途说明：
  - 对脂质注释字符串做标准化分类（适配多种类别），并提供提取碳数/不饱和度与安全文件名工具。

支持示例（不限于）：
    - Cer(d18:1/24:0)            -> Cer(d18:1)
    - CAEP(d18:2/27:4)(2OH)      -> CAEP(d18:2)(2OH)
    - FMC-4(d18:0/15:2)          -> FMC-4(d18:0)
    - GA2(d18:1/16:0)            -> GA2(d18:1)

规则：
    - 仅保留括号中“第一个片段”的骨架前缀与 X:Y（如 d18:1 -> d18:1），忽略斜杠后的第二片段；
  - 若在主括号后还有修饰括号（如 (2OH)、(dOH) 等），则原样保留在结果末尾；
  - 类别名为括号前文本，可包含字母、数字或连字符（如 CAEP、FMC-4、GA2、Cer 等）。

快速使用：
    >>> classify_lipid('CAEP(d18:2/27:4)(2OH)')
    'CAEP(d18:2)(2OH)'
  >>> extract_carbon_number('FMC-4(d18:0/15:2)')
  18
  >>> extract_unsaturation('GA2(d18:1/16:0)')
  1

输出：
    - classify_lipid: 标准化后的字符串，如 'CAEP(d18:2)(2OH)'，解析失败返回 '未分类'
  - extract_carbon_number: 第一个片段的碳数（int）或 None
  - extract_unsaturation: 第一个片段的不饱和度（int）或 None
  - sanitize_filename: 去除 Windows 非法字符后的安全文件名
    - extract_total_cn / extract_total_unsat / extract_total_cn_unsat: 从原始注释（含“/”的两段）提取“总碳数/总不饱和度”
----------------------------------------------------------
"""

import pandas as pd
import re


def _parse_annotation(annotation: str):
    """解析注释字符串，返回 (class_name, prefix, carbon, unsat, modifiers)

    兼容：'CAEP(d18:2/27:4)(2OH)'、'Cer(d14:0/20:0)'、'FMC-4(d18:0/15:2)'、'GA2(d18:1/16:0)'

    返回：
      - class_name: 括号前类别名（含数字/连字符）
      - prefix: 骨架前缀字母（如 d/t/m 等，保持原样大小写）
      - carbon: 第一个片段的碳数（int）
      - unsat: 第一个片段的不饱和度（int）
      - modifiers: 额外修饰（列表，形如 ['2OH', 'dOH']，来自主括号之后的括号）
    解析失败返回 None
    """
    if not isinstance(annotation, str) or pd.isna(annotation):
        return None

    # 匹配 类别名( 前缀X:Y [/ 任意... ] ) 后面可能还有多个修饰括号
    # 例： CAEP(d18:2/27:4)(2OH)
    main_pat = re.compile(r"^\s*([A-Za-z0-9\-]+)\s*\(\s*([A-Za-z]{1,3})(\d+):(\d+)(?:\s*/[^)]*)?\s*\)")
    m = main_pat.search(annotation)
    if not m:
        return None

    class_name = m.group(1)
    prefix = m.group(2)
    try:
        carbon = int(m.group(3))
        unsat = int(m.group(4))
    except Exception:
        return None

    # 提取主括号之后的修饰，如 (2OH)、(dOH) 等，全部原样保留
    tail = annotation[m.end():]
    mod_tokens = re.findall(r"\(([^()]+)\)", tail)
    # 仅保留非空的修饰字符串
    modifiers = [tok.strip() for tok in mod_tokens if tok.strip()]

    return class_name, prefix, carbon, unsat, modifiers


def classify_lipid(annotation: str) -> str:
    """标准化脂质注释为：Class(prefixX_Y)[(mods...)]

        示例：
            - 'Cer(d18:1/24:0)'         -> 'Cer(d18:1)'
            - 'CAEP(d18:2/27:4)(2OH)'   -> 'CAEP(d18:2)(2OH)'
            - 'FMC-4(d18:0/15:2)'       -> 'FMC-4(d18:0)'
            - 'GA2(d18:1/16:0)'         -> 'GA2(d18:1)'
    """
    parsed = _parse_annotation(annotation)
    if not parsed:
        return "未分类"

    class_name, prefix, carbon, unsat, modifiers = parsed
    base = f"{class_name}({prefix}{carbon}:{unsat})"
    if modifiers:
        # 原样拼接：(...)(...)
        for mod in modifiers:
            base += f"({mod})"
    return base


def extract_carbon_number(mother_ion: str):
    """提取第一个片段的碳数（X:Y 中的 X）。

    兼容任意类别名，形如 'XXX(d18:1/...)'。
    解析失败返回 None。
    """
    if not isinstance(mother_ion, str) or pd.isna(mother_ion):
        return None

    m = re.search(r"\(([A-Za-z]{1,3})(\d+):(\d+)", mother_ion)
    if not m:
        return None
    try:
        return int(m.group(2))
    except Exception:
        return None


def extract_unsaturation(mother_ion: str):
    """提取第一个片段的不饱和度（X:Y 中的 Y）。

    兼容任意类别名，形如 'XXX(d18:1/...)'。
    解析失败返回 None。
    """
    if not isinstance(mother_ion, str) or pd.isna(mother_ion):
        return None

    m = re.search(r"\(([A-Za-z]{1,3})(\d+):(\d+)\)", mother_ion)
    # 注：上面的正则要求右括号紧随其后；若存在 '/..'，则优先再宽松匹配：
    if not m:
        m = re.search(r"\(([A-Za-z]{1,3})(\d+):(\d+)", mother_ion)
    if not m:
        return None
    try:
        return int(m.group(3))
    except Exception:
        return None


def sanitize_filename(name: str) -> str:
    """清理字符串为合法文件名，替换或删除 Windows 不允许的字符。"""
    invalid_chars = ['\\\n', '\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name


# ---------------- 进阶：总碳数 & 总不饱和度提取 ----------------
def extract_total_cn_unsat(annotation: str):
    """从原始注释提取“总碳数、总不饱和度”。

    仅当原始注释中包含两段（形如 '... (dX:Y/A:B) ...'）时可计算：
      total_cn = X + A
      total_unsat = Y + B

    兼容第二段可省略骨架前缀（如 d14:2/12:3），也兼容存在前缀（如 d14:2/d12:3）。

    返回 (total_cn, total_unsat)；解析失败返回 (None, None)。
    """
    if not isinstance(annotation, str) or pd.isna(annotation):
        return (None, None)

    # 捕获括号内两段：第一段 prefixX:Y，第二段可写 A:B 或 prefixA:B
    # 分组：1=prefix1 2=X 3=Y 4=prefix2(可空) 5=A 6=B
    pat = re.compile(r"\(([A-Za-z]{1,3})(\d+):(\d+)/(?:([A-Za-z]{0,3})?)(\d+):(\d+)\)")
    m = pat.search(annotation)
    if not m:
        return (None, None)
    try:
        x = int(m.group(2)); y = int(m.group(3))
        a = int(m.group(5)); b = int(m.group(6))
    except Exception:
        return (None, None)
    return (x + a, y + b)


def extract_total_cn(annotation: str):
    """仅返回总碳数；解析失败返回 None。"""
    cn, _ = extract_total_cn_unsat(annotation)
    return cn


def extract_total_unsat(annotation: str):
    """仅返回总不饱和度；解析失败返回 None。"""
    _, u = extract_total_cn_unsat(annotation)
    return u
