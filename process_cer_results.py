import pandas as pd
import re
from pathlib import Path

# 仅保留占位分类：基名 + (d|t|m)x:y + 可选 0~2 个 (OH)/(2OH)/(dOH)

def classify_placeholder(annotation: str):
    if not isinstance(annotation, str) or pd.isna(annotation):
        return "未分类"
    m = re.match(r'^([^\(]+)\(([dtm])(\d+):(\d+)\)((?:\((?:2OH|dOH|OH)\)){0,2})', annotation)
    if not m:
        return "未分类"
    base = m.group(1).strip()
    dtm = m.group(2)
    mods = m.group(5)
    return f"{base}({dtm}x:y){mods}"

# 数字提取（仍保留总碳数/不饱和度及第一链信息）
TOTAL_PATTERN = re.compile(r'\([dtm](\d+):(\d+)\)')

def extract_total_numbers(annotation: str):
    if not isinstance(annotation, str) or pd.isna(annotation):
        return None, None
    m = TOTAL_PATTERN.search(annotation)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def extract_carbon_number(mother_ion):
    if not isinstance(mother_ion, str) or pd.isna(mother_ion):
        return None
    m = re.search(r'\([dtm](\d+):', mother_ion)
    return int(m.group(1)) if m else None

def extract_unsaturation(mother_ion):
    if not isinstance(mother_ion, str) or pd.isna(mother_ion):
        return None
    m = re.search(r'\([dtm]\d+:(\d+)\)', mother_ion)
    return int(m.group(1)) if m else None

def split_mz(value):
    """统一四位小数后拆分，返回 (整数部分, 小数部分[四位小数])."""
    try:
        v = float(value)
    except Exception:
        return None, None
    # 先标准化到四位小数
    v = round(v + 1e-12, 4)
    i = int(v)
    frac = v - i
    # 保留四位小数 单位变为mDa
    frac *= 1000
    return i, float(f"{frac:.1f}")

def process_excel(input_path: str, sheet_name: str | None = None, output_path: str | None = None,
                  type_column: str = '类型', mz_column: str = 'M+H'):
    input_path = input_path.strip("'\"")
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    df = pd.read_excel(in_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(in_path)

    if type_column not in df.columns:
        raise ValueError(f"未找到列: {type_column}")
    if mz_column not in df.columns:
        raise ValueError(f"未找到列: {mz_column}")

    # 唯一分类
    df['分类'] = df[type_column].apply(classify_placeholder)

    # 数字信息
    df['总碳数'], df['总不饱和度'] = zip(*df[type_column].apply(extract_total_numbers))
    # m/z 标准化与拆分
    df[mz_column] = df[mz_column].apply(lambda v: f"{float(v):.4f}" if pd.notna(v) else v)
    ints, fracs = zip(*df[mz_column].apply(split_mz))
    df['Integer mass'] = ints
    df['Decimal mass'] = fracs

    if not output_path:
        suffix = '_processed'
        output_path = in_path.with_stem(in_path.stem + suffix)
    out_path = Path(output_path)

    df.to_excel(out_path, index=False)
    print(f"处理完成 -> {out_path} (共 {len(df)} 行)")
    return out_path

if __name__ == '__main__':
    import sys
    import argparse

    # 如果命令行参数大于1个（除了脚本本身还有其他参数），则使用命令行模式
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='鞘脂结果Excel处理: 单一占位分类 + 数字提取 + m/z 拆分')
        parser.add_argument('input', help='输入Excel路径')
        parser.add_argument('-s', '--sheet', help='Sheet 名（不指定则读第一个）', default=None)
        parser.add_argument('-o', '--output', help='输出Excel路径（默认: 原名+_processed.xlsx）', default=None)
        parser.add_argument('--type-col', help='类型列名 (默认 类型)', default='类型')
        parser.add_argument('--mz-col', help='m/z 列名 (默认 M+H)', default='M+H')
        args = parser.parse_args()
        process_excel(args.input, args.sheet, args.output, args.type_col, args.mz_col)
    # 否则，进入为新手设计的交互模式
    else:
        print("--- 交互模式 ---")
        try:
            # 提示用户输入文件路径
            input_file = input("请输入Excel文件路径 (或将文件拖入终端后按回车): ").strip()
            # 去除路径两端可能存在的引号
            input_file = input_file.strip("'\"")
            
            if not input_file:
                print("错误：未输入路径，程序退出。")
            else:
                # 使用默认参数调用处理函数
                process_excel(input_file)
        except (KeyboardInterrupt, EOFError):
            print("\n操作已取消。")
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
