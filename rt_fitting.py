# -*- coding: utf-8 -*-
import os
import pandas as pd
import re
import numpy as np
import matplotlib
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from sklearn.linear_model import RANSACRegressor, LinearRegression
'''
linestyle=(0, (2.0, 3.0))  # 虚线样式示例
plt.scatter(..., s=100, ...) # 增大点的大小
plt.xlabel('Carbon Number', ..., fontsize=16)# 设置坐标轴字体大小
plt.title(..., fontsize=18)# 设置标题字体大小
'''
# 导入lipid_classifier模块
try:
    from lipid_classifier import (
        classify_lipid,
        extract_carbon_number,
        extract_unsaturation,
        sanitize_filename
    )
except ImportError as e:
    print("错误: 缺少lipid_classifier模块: {}".format(str(e)))
    sys.exit(1)

# 定义鲜艳的标准颜色列表（红、黄、蓝、绿、橙、粉、紫、青、棕、灰）
# 修改为模仿示意图的配色：蓝、绿、灰、橙、紫
colors = [
    '#5B9BD5',  # 蓝 (0)
    '#70AD47',  # 绿 (1)
    '#A5A5A5',  # 灰 (2)
    '#ED7D31',  # 橙 (3)
    '#6E2D9F',  # 紫 (4)
    '#8c564b',  # 棕
    '#e377c2',  # 粉
    '#bcbd22',  # 黄绿
    '#17becf',  # 青
    '#d62728',  # 红
]

def set_plot_style():
    """设置绘图风格"""
    # 尝试使用 Arial 字体，如果不可用则回退到 sans-serif
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300  # 提高分辨率
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.grid'] = False
    
    # 模仿示意图的边框风格
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

def linear_func(x, a, b):
    """线性函数: y = ax + b"""
    return a * x + b

def quad_func(x, a, b, c):
    """二次函数: y = ax² + bx + c"""
    return a * x ** 2 + b * x + c

def is_valid_quadratic_fit(params, x_range, quad_func):
    """
    判断二次拟合是否有效：顶点不在区间内，且区间内整体递增
    params: 二次拟合参数 (a, b, c)
    x_range: (x_min, x_max)
    quad_func: 二次函数
    返回: True/False
    """
    if params is None or len(params) != 3 or abs(params[0]) <= 1e-8:
        return False
    a, b, c = params
    x_min, x_max = x_range
    vertex_x = -b / (2 * a)
    # 顶点在区间内，判为无效
    if x_min <= vertex_x <= x_max:
        return False
    # 区间内整体递增（右端点预测值大于左端点预测值）
    y_left = quad_func(x_min, a, b, c)
    y_right = quad_func(x_max, a, b, c)
    if y_right <= y_left:
        return False
    return True

def fit_retention_time_curvefit_no_plot(x, y, unsat, color='b'):
    """
    对输入的总碳数和保留时间数据进行拟合（不生成图片）
    使用 RANSAC 算法进行鲁棒回归，以抵抗离群点和同分异构体的干扰。
    
    参数:
        x: 总碳数数组
        y: 保留时间数组
        unsat: 不饱和度值
        color: 颜色代码
        
    返回:
        r2: R平方值
        fit_type: 拟合类型（线性或二次）
        params: 拟合参数
        x_fit: 筛选后用于拟合的x值
        y_fit: 筛选后用于拟合的y值
        outliers: 剔除的离群点信息（如果有同分异构体，则包含其拟合结果）
    """
    x = np.array(x)
    y = np.array(y)
    
    # 格式化不饱和度为整数（去掉小数部分）
    unsat_int = int(float(unsat))
    
    # 初始化 outliers 结构
    outliers = {
        'x': [],
        'y': [],
        'is_isomer': False,
        'isomer_r2': 0,
        'isomer_fit_type': None,
        'isomer_params': None
    }

    # 检查数据点数量，少于3个无法拟合
    if len(x) <= 2:
        return 0, None, None, None, None, outliers

    # ---------------------------------------------------------
    # 使用 RANSAC 进行主趋势拟合
    # ---------------------------------------------------------
    
    X = x.reshape(-1, 1)
    
    # 定义 RANSAC 参数
    # residual_threshold=0.3 min. 
    # 对于保留时间，同分异构体通常偏差 > 0.5 min. 0.3 是一个相对严格但安全的阈值。
    # min_samples=3 (至少3点确定一条线，且能排除偶然)
    ransac = RANSACRegressor(random_state=42, min_samples=3, residual_threshold=0.4)
    
    try:
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
    except Exception as e:
        print(f"RANSAC 拟合失败: {e}")
        inlier_mask = np.ones(len(x), dtype=bool) # Fallback to use all
        outlier_mask = np.zeros(len(x), dtype=bool)

    # 提取 Inliers (主群)
    x_in = x[inlier_mask]
    y_in = y[inlier_mask]
    
    # 提取 Outliers (可能包含异构体)
    x_out = x[outlier_mask]
    y_out = y[outlier_mask]
    
    # 将 Outliers 存入结果
    outliers['x'] = x_out.tolist()
    outliers['y'] = y_out.tolist()
    
    # ---------------------------------------------------------
    # 对 Inliers 进行最终拟合 (Linear vs Quadratic)
    # ---------------------------------------------------------
    
    fit_type = None
    params = None
    r2 = 0
    x_fit = x_in
    y_fit = y_in
    
    if len(x_fit) >= 3:
        # 1. 尝试线性拟合
        try:
            popt_lin, _ = curve_fit(linear_func, x_fit, y_fit)
            y_pred_lin = linear_func(x_fit, *popt_lin)
            # 只有当样本数足够时才计算 R2，否则设为 0 或其他默认值
            if len(y_fit) > 1:
                r2_lin = r2_score(y_fit, y_pred_lin)
            else:
                r2_lin = 0
        except:
            r2_lin = -1; popt_lin = None
            
        # 2. 尝试二次拟合
        try:
             # 二次拟合至少需要 3 个点
             if len(x_fit) >= 3:
                 popt_quad, _ = curve_fit(quad_func, x_fit, y_fit)
                 y_pred_quad = quad_func(x_fit, *popt_quad)
                 if len(y_fit) > 1:
                     r2_quad = r2_score(y_fit, y_pred_quad)
                 else:
                     r2_quad = 0
             else:
                 r2_quad = -1; popt_quad = None
        except:
            r2_quad = -1; popt_quad = None
            
        # 判断有效性
        lin_valid = (r2_lin >= 0.99 and popt_lin is not None and popt_lin[0] > 0)
        
        quad_valid = False
        if r2_quad >= 0.99 and popt_quad is not None:
             quad_valid = is_valid_quadratic_fit(popt_quad, (np.min(x_fit), np.max(x_fit)), quad_func)
             
        # 决策逻辑：优先线性，除非二次显著更好或线性无效
        if lin_valid:
            if quad_valid and r2_quad > r2_lin + 0.005: # 稍微显著一点才选二次
                 fit_type = 'Filtered Quadratic'
                 params = popt_quad
                 r2 = r2_quad
            else:
                 fit_type = 'Filtered Linear'
                 params = popt_lin
                 r2 = r2_lin
        elif quad_valid:
             fit_type = 'Filtered Quadratic'
             params = popt_quad
             r2 = r2_quad
        else:
            fit_type = None
            params = None
            r2 = 0

    # 如果只有全量数据拟合且 RANSAC 没剔除任何点，改名
    if fit_type and len(x_fit) == len(x):
        fit_type = fit_type.replace('Filtered ', '')

    # ---------------------------------------------------------
    # 检查同分异构体 (Outliers)
    # ---------------------------------------------------------
    if len(x_out) >= 3:
        # 尝试线性拟合
        try:
            with np.errstate(all='ignore'):  # 忽略数值警告
                popt_out_lin, _ = curve_fit(linear_func, x_out, y_out)
                y_out_pred_lin = linear_func(x_out, *popt_out_lin)
                if len(y_out) > 1:
                    r2_out_lin = r2_score(y_out, y_out_pred_lin)
                else:
                    r2_out_lin = 0
        except Exception:
            r2_out_lin = 0; popt_out_lin = None
            
        # 尝试二次拟合
        try:
            with np.errstate(all='ignore'):  # 忽略数值警告
                 if len(x_out) >= 3:
                     popt_out_quad, _ = curve_fit(quad_func, x_out, y_out)
                     y_out_pred_quad = quad_func(x_out, *popt_out_quad)
                     if len(y_out) > 1:
                         r2_out_quad = r2_score(y_out, y_out_pred_quad)
                     else:
                        r2_out_quad = 0
                 else:
                    r2_out_quad = 0; popt_out_quad = None
        except Exception:
            r2_out_quad = 0; popt_out_quad = None
            
        # 二次有效性
        quad_vertex_valid = False
        if popt_out_quad is not None:
             quad_vertex_valid = is_valid_quadratic_fit(popt_out_quad, (np.min(x_out), np.max(x_out)), quad_func)
        
        # 判定
        found_isomer = False
        # 优先线性
        if r2_out_lin >= 0.99 and popt_out_lin is not None and popt_out_lin[0] > 0:
            outliers['isomer_r2'] = r2_out_lin
            outliers['isomer_fit_type'] = 'Isomer Linear'
            outliers['isomer_params'] = popt_out_lin
            found_isomer = True
        elif r2_out_quad >= 0.99 and quad_vertex_valid:
            outliers['isomer_r2'] = r2_out_quad
            outliers['isomer_fit_type'] = 'Isomer Quadratic'
            outliers['isomer_params'] = popt_out_quad
            found_isomer = True
            
        if found_isomer:
            outliers['is_isomer'] = True

    return r2, fit_type, params, x_fit, y_fit, outliers

def fit_retention_time_curvefit(x, y, category_name, unsat, output_dir, color='b', generate_plot=False):
    """
    使用无图版函数进行拟合，默认不生成单个图像，只返回拟合结果
    但可通过generate_plot参数控制是否生成单独的图
    
    参数:
        x: 总碳数数组
        y: 保留时间数组
        category_name: 类别名称
        unsat: 不饱和度值
        output_dir: 输出目录
        color: 颜色代码
        generate_plot: 是否生成单独的图表（默认为False）
        
    返回:
        r2: R平方值
        fit_type: 拟合类型
        params: 拟合参数
        isomer_info: 同分异构体信息（如果发现）
    """
    # 直接调用无图版函数进行拟合并获取结果
    r2, fit_type, params, x_fit, y_fit, outliers = fit_retention_time_curvefit_no_plot(x, y, unsat, color)
    
    # 只有在没有同分异构体时才生成单独的主拟合图表
    # (有同分异构体时会在下面生成组合图表)
    if generate_plot and fit_type and params is not None and r2 >= 0.99 and not outliers['is_isomer']:
        # 格式化不饱和度为整数
        unsat_int = int(float(unsat))
        
        # 新建画布
        plt.figure()
        x_curve = np.linspace(min(x_fit), max(x_fit), 100)
        
        # 严格区分线性/二次类型并绘制拟合曲线
        if 'Linear' in fit_type and len(params) == 2:
            y_curve = linear_func(x_curve, *params)
            plt.plot(x_curve, y_curve, color='#7f7f7f', linestyle='--', linewidth=2.5,
                     label=f'Unsat: {unsat_int} (Linear R^2={r2:.4f})')
        elif 'Quadratic' in fit_type and len(params) == 3:
            y_curve = quad_func(x_curve, *params)
            plt.plot(x_curve, y_curve, color='#7f7f7f', linestyle='--', linewidth=2.5,
                     label=f'Unsat: {unsat_int} (Quad R^2={r2:.4f})')
        
        # 绘制数据点，增强颜色饱和度和边框
        plt.scatter(x_fit, y_fit, color=color, alpha=1.0, s=100, 
                    edgecolors='none', label='Data Points')
        
        # 设置坐标轴标签和标题
        plt.xlabel('Carbon Number')
        plt.ylabel('$t_R$ (min)')
        plt.title(f'{category_name} - unsat {unsat_int} No Isomers')
        plt.legend()
        plt.tight_layout()
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 生成安全的文件名并保存图片
        safe_filename = sanitize_filename(f'{category_name}_{unsat_int}.png')
        plt.savefig(os.path.join(output_dir, safe_filename))
        plt.close()
    
    # 提取同分异构体信息
    isomer_info = None
    if outliers['is_isomer']:
        isomer_info = {
            'r2': outliers['isomer_r2'],
            'fit_type': outliers['isomer_fit_type'],
            'params': outliers['isomer_params'],
            'x': outliers['x'],
            'y': outliers['y'],
            'is_valid': True  # 默认认为是有效的同分异构体
        }
        
        # 检查主拟合和同分异构体的斜率差异，过大则认为是假阳性
        slope_diff_too_large = False
        
        # 获取主拟合的斜率或平均斜率
        main_slope = None
        if params is not None:
            if 'Linear' in fit_type and len(params) == 2:
                main_slope = params[0]  # 线性函数斜率就是a
            elif 'Quadratic' in fit_type and len(params) == 3 and x_fit is not None and len(x_fit) >= 2:
                # 对于二次函数，计算数据范围内的平均斜率
                a, b, c = params
                x_min, x_max = min(x_fit), max(x_fit)
                # 计算范围内的平均斜率：dy/dx在两端点的平均值
                slope_min = 2 * a * x_min + b  # f'(x_min)
                slope_max = 2 * a * x_max + b  # f'(x_max)
                main_slope = (slope_min + slope_max) / 2
        
        # 获取同分异构体的斜率或平均斜率
        isomer_slope = None
        if isomer_info['params'] is not None:
            if 'Linear' in isomer_info['fit_type'] and len(isomer_info['params']) == 2:
                isomer_slope = isomer_info['params'][0]  # 线性函数斜率就是a
            elif 'Quadratic' in isomer_info['fit_type'] and len(isomer_info['params']) == 3 and len(isomer_info['x']) >= 2:
                # 对于二次函数，计算数据范围内的平均斜率
                a, b, c = isomer_info['params']
                x_min, x_max = min(isomer_info['x']), max(isomer_info['x'])
                # 计算范围内的平均斜率
                slope_min = 2 * a * x_min + b  # f'(x_min)
                slope_max = 2 * a * x_max + b  # f'(x_max)
                isomer_slope = (slope_min + slope_max) / 2
        
        # 比较斜率差异
        if main_slope is not None and isomer_slope is not None:
            slope_diff = abs(main_slope - isomer_slope)
            
            if slope_diff > 0.6:
                print(f"  警告：同分异构体斜率({isomer_slope:.4f})与主拟合斜率({main_slope:.4f})差异过大({slope_diff:.4f})，可能为假阳性")
                slope_diff_too_large = True
                isomer_info['is_valid'] = False
        
        # 如果需要且拟合成功，且不是假阳性，生成包含主曲线和同分异构体的组合图表
        if (generate_plot or (isomer_info['r2'] >= 0.99 and r2 >= 0.99)) and not slope_diff_too_large:
            # 格式化不饱和度为整数
            unsat_int = int(float(unsat))
            
            # 新建画布
            plt.figure(figsize=(9, 6))
            
            # 1. 绘制主要拟合曲线（如果拟合成功）
            if r2 >= 0.99 and fit_type and params is not None and x_fit is not None and y_fit is not None:
                # 绘制主拟合的数据点
                plt.scatter(x_fit, y_fit, color='#1f77b4', alpha=1.0, s=100, 
                           edgecolors='none', label='Main Group')
                
                x_curve_main = np.linspace(min(x_fit), max(x_fit), 100)
                if 'Linear' in fit_type and len(params) == 2:
                    # 再次验证斜率为正
                    if params[0] >= 0:
                        y_curve_main = linear_func(x_curve_main, *params)
                        plt.plot(x_curve_main, y_curve_main, color='#7f7f7f', linestyle='--', linewidth=2.5,
                                label=f'Main Fit (Linear $R^2$={r2:.4f})')
                    else:
                        print(f"  警告：主拟合线性拟合斜率为负，跳过绘图")
                elif 'Quadratic' in fit_type and len(params) == 3:
                    # 再次验证顶点不在区间内
                    a, b, c = params
                    if abs(a) > 1e-8:
                        vertex_x = -b / (2 * a)
                        x_min, x_max = min(x_fit), max(x_fit)
                        if not (x_min <= vertex_x <= x_max):
                            y_curve_main = quad_func(x_curve_main, *params)
                            plt.plot(x_curve_main, y_curve_main, color='#7f7f7f', linestyle='--', linewidth=2.5,
                                    label=f'Main Fit (Quad $R^2$={r2:.4f})')
                        else:
                            print(f"  警告：主拟合二次拟合顶点在数据范围内，跳过绘图")
                    else:
                        y_curve_main = quad_func(x_curve_main, *params)
                        plt.plot(x_curve_main, y_curve_main, color='#7f7f7f', linestyle='--', linewidth=2.5,
                                label=f'Main Fit (Quad $R^2$={r2:.4f})')
            
            # 2. 绘制同分异构体拟合曲线（如果拟合成功）
            if isomer_info['r2'] >= 0.99:
                # 绘制同分异构体数据点
                plt.scatter(isomer_info['x'], isomer_info['y'], color='#d62728', alpha=1.0, s=100, 
                           edgecolors='none', label='Isomer Group')
                
                x_curve_isomer = np.linspace(min(isomer_info['x']), max(isomer_info['x']), 100)
                if 'Linear' in isomer_info['fit_type'] and len(isomer_info['params']) == 2:
                    # 再次验证斜率为正
                    if isomer_info['params'][0] >= 0:
                        y_curve_isomer = linear_func(x_curve_isomer, *isomer_info['params'])
                        plt.plot(x_curve_isomer, y_curve_isomer, color='#d62728', linestyle='--', linewidth=2.5,
                                label=f'Isomer Fit (Linear $R^2$={isomer_info["r2"]:.4f})')
                    else:
                        print(f"  警告：同分异构体线性拟合斜率为负，跳过绘图")
                elif 'Quadratic' in isomer_info['fit_type'] and len(isomer_info['params']) == 3:
                    # 再次验证顶点不在区间内
                    a, b, c = isomer_info['params']
                    if abs(a) > 1e-8:
                        vertex_x = -b / (2 * a)
                        x_min, x_max = min(isomer_info['x']), max(isomer_info['x'])
                        if not (x_min <= vertex_x <= x_max):
                            y_curve_isomer = quad_func(x_curve_isomer, *isomer_info['params'])
                            plt.plot(x_curve_isomer, y_curve_isomer, color='#d62728', linestyle='--', linewidth=2.5,
                                    label=f'Isomer Fit (Quad $R^2$={isomer_info["r2"]:.4f})')
                        else:
                            print(f"  警告：同分异构体二次拟合顶点在数据范围内，跳过绘图")
                    else:
                        y_curve_isomer = quad_func(x_curve_isomer, *isomer_info['params'])
                        plt.plot(x_curve_isomer, y_curve_isomer, color='#d62728', linestyle='--', linewidth=2.5,
                                label=f'Isomer Fit (Quad $R^2$={isomer_info["r2"]:.4f})')
            
            # 3. 绘制剩余未使用的原始数据点（如果有）
            unused_points_x = []
            unused_points_y = []
            for i in range(len(x)):
                is_in_main = False
                if x_fit is not None:
                    for j in range(len(x_fit)):
                        if abs(x[i] - x_fit[j]) < 1e-6 and abs(y[i] - y_fit[j]) < 1e-6:
                            is_in_main = True
                            break
                
                is_in_isomer = False
                for j in range(len(isomer_info['x'])):
                    if abs(x[i] - isomer_info['x'][j]) < 1e-6 and abs(y[i] - isomer_info['y'][j]) < 1e-6:
                        is_in_isomer = True
                        break
                
                if not is_in_main and not is_in_isomer:
                    unused_points_x.append(x[i])
                    unused_points_y.append(y[i])
            
            if unused_points_x:
                plt.scatter(unused_points_x, unused_points_y, color='gray', alpha=0.5, s=25, 
                           marker='x', label='Unused Points')
            
            # 设置图表属性
            plt.xlabel('Carbon Number')
            plt.ylabel('$t_R$ (min)')
            plt.title(f'{category_name} - unsat {unsat_int} Fitting and Isomers')
            plt.grid(False)
            # 强制横坐标只显示整数
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, borderpad=0.4)
            plt.tight_layout()
            
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 生成安全的文件名并保存图片
            safe_filename = sanitize_filename(f'{category_name}_{unsat_int}_combined.png')
            plt.savefig(os.path.join(output_dir, safe_filename))
            plt.close()
    
    # 返回拟合结果和同分异构体信息
    return r2, fit_type, params, isomer_info

def plot_all_unsaturations(category_data, category_name, output_dir):
    """
    绘制同一分类下所有不饱和度的拟合曲线的总览图
    只生成一个总览图，不生成每个不饱和度的单独图表

    参数:
        category_data: 包含某一分类所有数据的DataFrame
        category_name: 类别名称
        output_dir: 输出目录
        
    注意：
        现在已优化为默认只生成总览图，不生成单独的不饱和度图表
        如需单独图表，可调用fit_retention_time_curvefit时将generate_plot参数设为True
    """
    set_plot_style()
    color_idx = 0
    has_data = False
    plt.figure(figsize=(7, 5))  # 创建总览图画布（含虚点版本）- 同步尺寸 (9, 6) -> (7, 5)
    ax = plt.gca()
    # 强制横坐标只显示整数
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # 同步tick_params
    ax.tick_params(axis='both', which='both', direction='out', colors='black', labelsize=14, length=6, width=1.5)

    # 存储所有不饱和度的拟合结果
    individual_results = []
    
    # 第一步：为每个不饱和度单独获取拟合结果（不生成单独的图）
    valid_fits_count = 0 
    for unsat, group in category_data.groupby('Unsaturation'):
        # 不饱和度格式化为整数
        unsat_int = int(float(unsat))
        
        if len(group) < 2:
            continue
            
        x = group['TotalCarbons'].values
        y = group['RT'].values
        
        # 只获取拟合结果，不生成单独的图
        r2, fit_type, params, x_fit, y_fit, _ = fit_retention_time_curvefit_no_plot(
            x, y, unsat, color=colors[color_idx % len(colors)]
        )
        
        if r2 >= 0.99 and fit_type and params is not None:
             valid_fits_count += 1
        
        # 记录拟合结果（无论是否达标）
        individual_results.append({
            'unsat': unsat,
            'unsat_int': unsat_int,
            'x': x,  # 原始数据点
            'y': y,  # 原始数据点
            'x_fit': x_fit,  # 拟合后的数据点（可能过滤了离群值）
            'y_fit': y_fit,  # 拟合后的数据点（可能过滤了离群值）
            'r2': r2,
            'fit_type': fit_type,
            'params': params,
            'color_idx': color_idx
        })
        
        color_idx += 1
        has_data = True
    
    # 如果没有数据，或者没有任意一个有效拟合结果，关闭图形并返回
    # 用户要求：如果没有拟合成功就不要输出图了
    if not has_data or valid_fits_count == 0:
        plt.close()
        return
        
    # 第二步：在总览图中绘制所有不饱和度的数据点和拟合曲线（含虚点版本）
    for result in individual_results:
        unsat_int = result['unsat_int']
        x = result['x']  # 原始数据点
        y = result['y']  # 原始数据点
        x_fit = result['x_fit']  # 拟合使用的数据点（可能剔除了离群点）
        y_fit = result['y_fit']  # 拟合使用的数据点（可能过滤了离群值）
        r2 = result['r2']
        fit_type = result['fit_type']
        params = result['params']
        color_idx = result['color_idx'] % len(colors)
        color = colors[color_idx]
        
        # 原始数据（虚点）：不进入图例
        plt.scatter(
            x, y, 
            color=color, 
            alpha=0.25,
            s=80, 
            edgecolors='none',
            linewidths=0.0, 
            label='_nolegend_'
        )
        
        # 拟合使用的数据点（实点）：图例只显示不饱和度数字
        if x_fit is not None and y_fit is not None:
            plt.scatter(
                x_fit, y_fit,
                color=color, 
                alpha=1.0,
                s=100,  # 同步大小 120 -> 140
                edgecolors='none', 
                label=f'{unsat_int}'
            )

        # 拟合曲线（若成功）：不单独加入图例
        if fit_type and params is not None and r2 >= 0.99:
            x_curve = np.linspace(min(x_fit) if x_fit is not None else min(x), 
                                max(x_fit) if x_fit is not None else max(x), 100)
            if 'Linear' in fit_type and len(params) == 2 and params[0] >= 0:
                y_curve = linear_func(x_curve, *params)
                plt.plot(
                    x_curve,
                    y_curve,
                    color=color,
                    linestyle=(0, (2.0, 3.0)), # 同步线型 (0, (1.5, 1)) -> (0, (2, 3))
                    linewidth=2.0,
                    label='_nolegend_'
                )
            elif 'Quadratic' in fit_type and len(params) == 3:
                a, b, c = params
                draw_curve = True
                if abs(a) > 1e-8 and x_fit is not None:
                    vertex_x = -b / (2 * a)
                    x_min, x_max = min(x_fit), max(x_fit)
                    if x_min <= vertex_x <= x_max:
                        draw_curve = False
                if draw_curve:
                    y_curve = quad_func(x_curve, *params)
                    plt.plot(
                        x_curve,
                        y_curve,
                        color=color,
                        linestyle=(0, (2.0, 3.0)), # 同步线型
                        linewidth=2.0,
                        label='_nolegend_'
                    )

    if has_data:
        plt.xlabel('Carbon Number', fontweight='bold', fontsize=16) # fontsize 14 -> 16
        plt.ylabel('$t_R$ (min)', fontweight='bold', fontsize=16) # fontsize 14 -> 16
        plt.title('{}'.format(category_name), fontweight='bold', fontsize=18) # fontsize 16 -> 18
        # 加粗刻度
        ax = plt.gca()
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
        
        # 模仿示意图的图例风格：左上角，横向或紧凑的框
        handles, labels = plt.gca().get_legend_handles_labels()
        seen = set()
        new_h, new_l = [], []
        # 重新排序图例，确保按 1, 2, 3, 4 顺序
        sorted_legends = sorted(zip(handles, labels), key=lambda x: x[1])
        
        for h, l in sorted_legends:
            if l != '_nolegend_' and l not in seen:
                seen.add(l)
                new_h.append(h)
                new_l.append(l)
        
        if new_h:
            # 使用 handletextpad 和 borderpad 调整图例紧凑度
            legend = plt.legend(new_h, new_l, frameon=True, fontsize=12, 
                               loc='upper left', ncol=len(new_h), 
                               handletextpad=0.1, columnspacing=0.5,
                               edgecolor='black', fancybox=False)
            legend.get_frame().set_linewidth(0.8)
            
        plt.tight_layout()
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        safe_filename = sanitize_filename('{}_all.png'.format(category_name))
        plt.savefig(os.path.join(output_dir, safe_filename))
        plt.close()
    else:
        plt.close()

    # === 生成“无虚点”版本：只包含拟合成功的不饱和度（实点+拟合曲线），图例为竖排数字 ===
    set_plot_style()
    plt.figure(figsize=(7, 5)) # 调整尺寸，使其更紧凑 (原9,6 -> 7,5)
    ax2 = plt.gca()
    # 强制横坐标只显示整数
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # labelsize从16 降到 14
    ax2.tick_params(axis='both', which='both', direction='out', colors='black', labelsize=14, length=6, width=1.5)

    for result in individual_results:
        unsat_int = result['unsat_int']
        x = result['x']
        y = result['y']
        x_fit = result['x_fit']
        y_fit = result['y_fit']
        r2 = result['r2']
        fit_type = result['fit_type']
        params = result['params']
        color_idx = result['color_idx'] % len(colors)
        color = colors[color_idx]

        # 仅当拟合成功时才绘制该不饱和度（隐藏未拟合成功的组）
        if not (fit_type and params is not None and r2 >= 0.99 and x_fit is not None and y_fit is not None):
            continue

        # 绘制拟合点（实点）
        plt.scatter(
            x_fit, y_fit,
            color=color,
            alpha=1.0, # 不透明
            s=100, # 增大点的大小 (120 -> 160 -> 140)
            edgecolors='none', # 去掉黑色边框
            label=f'{unsat_int}'
        )

        # 曲线 - 使用更密集的虚线
        x_curve = np.linspace(min(x_fit), max(x_fit), 100)
        draw_curve = False
        if 'Linear' in fit_type and len(params) == 2 and params[0] >= 0:
            y_curve = linear_func(x_curve, *params)
            draw_curve = True
        elif 'Quadratic' in fit_type and len(params) == 3:
            a, b, c = params
            draw_curve_quad = True
            if abs(a) > 1e-8:
                vertex_x = -b / (2 * a)
                x_min, x_max = min(x_fit), max(x_fit)
                if x_min <= vertex_x <= x_max:
                    draw_curve_quad = False
            if draw_curve_quad:
                y_curve = quad_func(x_curve, *params)
                draw_curve = True
        
        if draw_curve:
            # linestyle=(0, (2.0, 3.0)) 稍微稀疏一点的点划线
            plt.plot(x_curve, y_curve, color=color, linestyle=(0, (2.0, 3.0)), linewidth=2.0, label='_nolegend_')

    plt.xlabel('Carbon Number', fontweight='bold', fontsize=16) # fontsize 18 -> 16
    plt.ylabel('$t_R$ (min)', fontweight='bold', fontsize=16) # fontsize 18 -> 16
    # 标题不写“亚类拟合（无虚点）”，仅分类名
    plt.title('{}'.format(category_name), fontweight='bold', fontsize=18) # fontsize 20 -> 18
    # 加粗刻度
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_fontweight('bold')
        
    # 图例风格：左上角，横向 (调整得更紧凑)
    handles2, labels2 = plt.gca().get_legend_handles_labels()
    seen2 = set()
    new_h2, new_l2 = [], []
    sorted_legends2 = sorted(zip(handles2, labels2), key=lambda x: x[1])
    
    for h, l in sorted_legends2:
        if l != '_nolegend_' and l not in seen2:
            seen2.add(l)
            new_h2.append(h)
            try:
                # 尝试将标签转为数字以排序
                new_l2.append(int(l))
            except:
                new_l2.append(l)

    # 再次按数字排序
    final_legends = sorted(zip(new_h2, new_l2), key=lambda x: str(x[1]))
    final_h = [x[0] for x in final_legends]
    final_l = [str(x[1]) for x in final_legends]

    if final_h:
        legend = plt.legend(final_h, final_l, frameon=True, fontsize=12, 
                           loc='upper left', ncol=len(final_h),
                           handletextpad=0.1, columnspacing=0.5,
                           edgecolor='black', fancybox=False)
        legend.get_frame().set_linewidth(0.8)
        
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    safe_filename2 = sanitize_filename('{}_all_clean.png'.format(category_name))
    plt.savefig(os.path.join(output_dir, safe_filename2))
    plt.close()

def rename_dataframe_columns_to_english(df):
    """
    将中文列名重命名为英文，以增强兼容性
    
    参数:
        df: 输入的DataFrame
        
    返回:
        重命名列后的DataFrame
    """
    column_map = {
        '不饱和度': 'Unsaturation',
        '总碳数': 'TotalCarbons',
        '分类': 'Category',
        '保留时间': 'RT',
        '丰度': 'Abund'
    }
    
    # 只重命名存在且需要重命名的列
    rename_dict = {col: column_map[col] for col in df.columns if col in column_map}
    if rename_dict:
        return df.rename(columns=rename_dict)
    return df

def main():
    """模块主函数，仅在直接执行此脚本时调用"""
    print("这是用于保留时间拟合的实用程序模块。")
    print("请在您的主脚本中导入这些函数使用。")

if __name__ == "__main__":
    main()
    plt.close('all')  # 强制关闭所有未关闭的figure，彻底防止内存泄漏和警告
