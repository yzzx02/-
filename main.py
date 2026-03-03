# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import os
import sys
''' '''
# 尝试导入可能不存在的包并提供友好的错误消息
try:
    import matplotlib.pyplot as plt
    import matplotlib
    from sklearn.metrics import r2_score
    from scipy.optimize import curve_fit
    
    # 导入分类相关的函数
    from lipid_classifier import (
        classify_lipid,
        extract_carbon_number,
        extract_unsaturation,
        sanitize_filename
    )
    
    # 从优化的绘图模块导入函数
    from rt_fitting import (
        set_plot_style, 
        fit_retention_time_curvefit, 
        plot_all_unsaturations,
        plot_single_unsaturation_group
    )
except ImportError as e:
    print(f"错误: 缺少必要的依赖包: {str(e)}")
    print("请尝试运行 'pip install matplotlib numpy scipy scikit-learn pandas' 安装所需包")
    sys.exit(1)

# 基础函数定义
def linear_func(x, a, b):
    return a * x + b

def quad_func(x, a, b, c):
    return a * x ** 2 + b * x + c

def main():
    try:
        # 读取Excel文件（使用固定路径进行测试）
        file_path = input("请输入Excel文件路径（例如 'RT结果.xlsx'）: ").strip().strip('"\'')
        df = pd.read_excel(file_path)

        # ---- 列名规范化：去除隐含空格/全角空格，统一常见同义列名 ----
        def _norm(s: str) -> str:
            s = str(s)
            # 去除普通空格与不间断空格，统一小写
            s = s.replace('\u00A0', '').replace('\u3000', '').strip()
            return re.sub(r"\s+", "", s).lower()

        def _pick_col(df_cols, candidates):
            cand_norm = [_norm(c) for c in candidates]
            for c in df_cols:
                cn = _norm(c)
                if cn in cand_norm:
                    return c
            return None

        # 原始列名列表
        orig_cols = list(df.columns)
        # 先做一次简单去空白重命名（不改变语义，仅修剪）
        cleaned = {}
        for c in orig_cols:
            c_new = c
            if isinstance(c_new, str):
                c_new = c_new.replace('\u00A0', '').replace('\u3000', '').strip()
            cleaned[c] = c_new
        if cleaned:
            df = df.rename(columns=cleaned)

        # 统一关键列名：注释 / 母离子 / RT
        col_map = {}
        ann_col = _pick_col(df.columns, ['注释', 'annotation', '名称', 'name'])
        if ann_col and ann_col != '注释':
            col_map[ann_col] = '注释'

        parent_col = _pick_col(
            df.columns,
            ['母离子', '母离子注释', '母离子名称', '母离子信息', '母离子(注释)', '母离子（注释）', '母 离子']
        )
        if parent_col and parent_col != '母离子':
            col_map[parent_col] = '母离子'

        rt_col = _pick_col(
            df.columns,
            ['RT', '保留时间', '保留时间(min)', '保留时间(二元泵)', '保留时间归一化（二元泵）', 'rt(min)', 'rt']
        )
        if rt_col and rt_col != 'RT':
            col_map[rt_col] = 'RT'

        if col_map:
            df = df.rename(columns=col_map)
            print(f"列名已标准化: {col_map}")
        
        # 检查是否有"注释"和"分类"列，决定是否需要分类
        if "分类" in df.columns:
            print("警告：Excel文件中已经有'分类'列，将跳过分类步骤")
        elif "分类" not in df.columns and "注释" in df.columns:
            # 如果没有分类列但有注释列，则进行分类
            df["分类"] = df["注释"].apply(classify_lipid)
            print("分类已完成")
        else:
            print("错误：Excel文件中没有'注释'列")
            return
        
        # 检查是否有"母离子"列，决定是否可以提取碳数和不饱和度；没有则回退用“注释”列解析
        if "母离子" in df.columns:
            df["总碳数"] = df["母离子"].apply(extract_carbon_number)
            df["不饱和度"] = df["母离子"].apply(extract_unsaturation)
        elif "注释" in df.columns:
            print("提示：未找到'母离子'列，改用'注释'列解析碳数与不饱和度")
            df["总碳数"] = df["注释"].apply(extract_carbon_number)
            df["不饱和度"] = df["注释"].apply(extract_unsaturation)
        else:
            print("警告：Excel文件中没有'母离子'列，也没有'注释'列，无法提取碳数和不饱和度")
        
        # 保存处理后的Excel文件
        output_path = file_path.replace('.xlsx', '_分类结果.xlsx')
        if file_path == output_path:
            output_path = file_path.replace('.xls', '_分类结果.xlsx')
        try:
            df.to_excel(output_path, index=False)
        except UnicodeEncodeError:
            # 处理编码问题
            print("警告：遇到编码问题，尝试使用UTF-8编码保存")
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
        print(f"分类完成，结果已保存至: {output_path}")
        
        # 显示分类统计信息
        class_counts = df["分类"].value_counts()
        print("\n分类统计:")
        for category, count in class_counts.items():
            print(f"{category}: {count}个")
        
        # 创建拟合结果输出目录
        output_dir = os.path.join(os.path.dirname(file_path), "拟合结果")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 初始化拟合结果列表
        fit_results = []
        
        # 按分类分组，逐类处理
        for category, category_data in df.groupby('分类'):
            print(f"\n处理分类: {category}")
            # 重命名列为英文，便于后续处理
            category_data_en = category_data.rename(columns={
                '不饱和度': 'Unsaturation',
                '总碳数': 'TotalCarbons'
            })
            # 绘制该分类下所有不饱和度的总览拟合图（只画总图，不画单图）
            plot_all_unsaturations(category_data_en, category, output_dir)
            
            # 按不饱和度分组，逐个拟合
            for unsat, group in category_data_en.groupby('Unsaturation'):
                unsat_int = int(float(unsat))
                print(f"  不饱和度: {unsat_int}, 数据点数: {len(group)}")
                # 数据点数量不足则跳过
                if len(group) < 2:
                    print(f"  跳过拟合，数据点数量不足")
                    continue
                # 筛选碳数区间（25-50），保证拟合数据合理
                filtered_group = group
                if len(filtered_group) < 2:
                    print(f"跳过拟合，筛选后的数据点数量不足")
                    continue
                # 拟合前再检查数据点数量
                x = filtered_group['TotalCarbons'].values  # x轴：总碳数
                y = filtered_group['RT'].values    # y轴：保留时间
                if len(x) <= 2:
                    print(f"  跳过拟合，数据点数量不足（至少需要3个点）")
                    continue
                # 为该(分类, 不饱和度)组合生成单独图表
                plot_single_unsaturation_group(x, y, category, unsat_int, output_dir)
                # 调用拟合函数（默认不生成单图）
                r2, fit_type, params, isomer_info = fit_retention_time_curvefit(x, y, category, unsat, output_dir)
                if fit_type is None:
                    print(f"  拟合失败，可能是数据点特征不适合拟合")
                    continue
                
                # 如果发现同分异构体，输出信息
                if isomer_info is not None:
                    isomer_r2 = isomer_info['r2']
                    isomer_fit_type = isomer_info['fit_type']
                    is_valid_isomer = isomer_info.get('is_valid', True)  # 获取同分异构体是否有效的标记
                    
                    if not is_valid_isomer:
                        print(f"  发现潜在同分异构体，但被判定为假阳性，已忽略")
                        continue
                        
                    print(f"  发现同分异构体: {isomer_fit_type}, R²: {isomer_r2:.4f}")
                    
                    # 注意：rt_fitting.py 内部逻辑已经会自动生成同分异构体分析图（只要R²满足条件）
                    # 因此无需再次调用 fit_retention_time_curvefit
                    if isomer_r2 >= 0.99 and r2 >= 0.99:
                         print(f"  已自动生成同分异构体分析图: {category}_{unsat_int}_isomer_analysis.png")
                    
                    # 记录同分异构体拟合结果
                    if isomer_r2 >= 0.99:
                        if 'Quadratic' in isomer_fit_type and len(isomer_info['params']) == 3:
                            a, b, c = isomer_info['params']
                            equation = f"y = {a:.4f}x² + {b:.4f}x + {c:.4f}"
                        elif 'Linear' in isomer_fit_type and len(isomer_info['params']) == 2:
                            equation = f"y = {isomer_info['params'][0]:.4f}x + {isomer_info['params'][1]:.4f}"
                        else:
                            equation = "拟合方程类型错误"
                            
                        fit_results.append({
                            '分类': f"{category} (同分异构体)",
                            '不饱和度': unsat_int,
                            '数据点数': len(isomer_info['x']),
                            '拟合类型': isomer_fit_type,
                            'R²': isomer_r2,
                            '方程': equation
                        })
                
                # 拟合达标（R²>=0.99）才记录结果
                if r2 >= 0.99:
                    # 生成拟合方程字符串
                    if fit_type and ('Quadratic' in fit_type) and len(params) == 3:
                        a, b, c = params
                        x_min, x_max = min(x), max(x)
                        x0 = -b / (2 * a) if a != 0 else None
                        # 过滤规则：a在-1到1之间，且顶点不能在x区间内
                        if not (-1 <= a <= 1) or (x0 is not None and x_min < x0 < x_max):
                            x0_str = f"{x0:.2f}" if x0 is not None else "None"
                            print(f"  二次拟合被过滤：a={a:.4f}, 顶点x0={x0_str} 在区间[{x_min},{x_max}]，不保留该结果")
                            continue
                        equation = f"y = {a:.4f}x² + {b:.4f}x + {c:.4f}"
                        # if 'Filtered' in fit_type:
                        #     # equation += " (过滤后)"
                    elif fit_type and ('Linear' in fit_type) and len(params) == 2:
                        equation = f"y = {params[0]:.4f}x + {params[1]:.4f}"
                        # if 'Filtered' in fit_type:
                        #     # equation += " (过滤后)"
                    else:
                        equation = f"拟合方程类型错误"
                        continue
                    # 记录拟合结果到列表
                    fit_results.append({
                        '分类': category,
                        '不饱和度': unsat_int,
                        '数据点数': len(filtered_group),
                        '拟合类型': fit_type,
                        'R²': r2,
                        '方程': equation
                    })
                    print(f"  拟合类型: {fit_type}, R²: {r2:.4f}")
                    print(f"  拟合方程: {equation}")
                else:
                    print(f"  R²未达标（{r2:.4f}），不记录该结果")
        # 拟合结果保存到Excel
        if fit_results:
            fit_df = pd.DataFrame(fit_results)
            fit_results_path = os.path.join(output_dir, "拟合结果汇总.xlsx")
            try:
                fit_df.to_excel(fit_results_path, index=False)
            except UnicodeEncodeError:
                # 处理编码问题
                print("警告：遇到编码问题，尝试使用UTF-8编码保存拟合结果")
                with pd.ExcelWriter(fit_results_path, engine='openpyxl') as writer:
                    fit_df.to_excel(writer, index=False)
            print(f"\n拟合结果已保存至: {fit_results_path}")
        else:
            print("\n没有找到足够的数据进行拟合")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    plt.close('all')  # 强制关闭所有未关闭的figure，彻底防止内存泄漏和警告