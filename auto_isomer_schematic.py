import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import re
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ==========================================
# 核心逻辑 (复刻 rt_fitting.py)
# ==========================================

def linear_func(x, a, b):
    """线性函数: y = ax + b"""
    return a * x + b

def quad_func(x, a, b, c):
    """二次函数: y = ax² + bx + c"""
    return a * x ** 2 + b * x + c

def is_valid_quadratic_fit(params, x_range, quad_func):
    if params is None or len(params) != 3 or abs(params[0]) <= 1e-8:
        return False
    a, b, c = params
    x_min, x_max = x_range
    vertex_x = -b / (2 * a)
    if x_min <= vertex_x <= x_max:
        return False
    y_left = quad_func(x_min, a, b, c)
    y_right = quad_func(x_max, a, b, c)
    if y_right <= y_left: # 应该单调递增
        return False
    return True

def analyze_lipid_series(x_input, y_input):
    """
    自动分析脂质序列，识别主群和同分异构体
    """
    x = np.array(x_input)
    y = np.array(y_input)
    
    # 定义拟合函数
    def fit_linear(x, y):
        try:
            if len(y) < 2:
                return 0, None, None
            popt, _ = curve_fit(linear_func, x, y)
            y_pred = linear_func(x, *popt)
            if len(y) > 1:
                r2 = r2_score(y, y_pred)
            else:
                r2 = 0
            return r2, popt, y_pred
        except:
            return 0, None, None

    def fit_quad(x, y):
        try:
            if len(y) < 3:
                return 0, None, None
            popt, _ = curve_fit(quad_func, x, y, p0=[0.01, 0.1, 0], bounds=([-1, -np.inf, -np.inf], [1, np.inf, np.inf]), maxfev=5000)
            y_pred = quad_func(x, *popt)
            if len(y) > 1:
                r2 = r2_score(y, y_pred)
            else:
                r2 = 0
            return r2, popt, y_pred
        except:
            return 0, None, None

    # 初始化
    x_fit = x.copy()
    y_fit = y.copy()
    
    outliers = {
        'x': [],
        'y': []
    }
    
    # 迭代剔除离群点
    fit_result_main = {}
    
    while len(x_fit) > 3:
        r2_lin, popt_lin, y_pred_lin = fit_linear(x_fit, y_fit)
        
        # 简单起见，主要用线性拟合作为判断基准 (脂质通常是线性的)
        # 如果需要二次，可以扩展
        
        lin_valid = (r2_lin >= 0.99)
        
        if lin_valid:
            fit_result_main = {
                'type': 'Linear',
                'params': popt_lin,
                'r2': r2_lin,
                'func': linear_func,
                'x_fit': x_fit,
                'y_fit': y_fit
            }
            break
            
        # 剔除残差最大的点
        if y_pred_lin is not None:
             residuals = np.abs(y_fit - y_pred_lin)
             idx = np.argmax(residuals)
             
             outliers['x'].append(x_fit[idx])
             outliers['y'].append(y_fit[idx])
             
             x_fit = np.delete(x_fit, idx)
             y_fit = np.delete(y_fit, idx)
        else:
            break
            
    # 如果循环结束还没找到完美的，就用最后剩下的拟合一下
    if not fit_result_main and len(x_fit) >= 2:
         r2_lin, popt_lin, y_pred_lin = fit_linear(x_fit, y_fit)
         fit_result_main = {
                'type': 'Linear',
                'params': popt_lin,
                'r2': r2_lin,
                'func': linear_func,
                'x_fit': x_fit,
                'y_fit': y_fit
        }

    # 分析 Outliers 是否构成同分异构体
    fit_result_isomer = None
    if len(outliers['x']) >= 3:
        x_out = np.array(outliers['x'])
        y_out = np.array(outliers['y'])
        
        # 对 outliers 进行拟合
        r2_iso, popt_iso, _ = fit_linear(x_out, y_out)
        
        if r2_iso > 0.98: # 同分异构体判定阈值
             fit_result_isomer = {
                'type': 'Linear',
                'params': popt_iso,
                'r2': r2_iso,
                'func': linear_func,
                'x_fit': x_out,
                'y_fit': y_out
            }

    return fit_result_main, outliers, fit_result_isomer

def sanitize_filename(name):
    """
    清理文件名，移除非法字符 (Windows: < > : " / \ | ? *)
    """
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def plot_two_steps(name, x_all, y_all, main_res, outliers_data, isomer_res, output_dir='示意图生成', std_data=None):
    # 转换为绝对路径，确保知道文件去哪了
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = os.path.join(script_dir, output_dir)
    
    if not os.path.exists(output_dir_abs):
        os.makedirs(output_dir_abs)
    
    # 清理文件名
    safe_name = sanitize_filename(name)

    # 样式设置
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.direction'] = 'out' 
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['figure.dpi'] = 300
    
    # 优化的配色方案 (文献级)
    color_main = '#3B75AF'      # Classic Blue (Seaborn Deep)
    color_isomer = '#D64E52'    # Muted Red (Seaborn Deep)
    color_std = "#b95ac5"       # Purple (Muted)
    color_line_main = '#7f7f7f' # Grey
    color_line_isomer = '#D64E52' # Same as isomer points
    
    x_min = min(x_all) - 1
    x_max = max(x_all) + 1
    x_line = np.linspace(x_min, x_max, 100)
    
    y_min, y_max = min(y_all), max(y_all)
    y_range = y_max - y_min
    if y_range == 0: y_range = 1.0
    y_lim = (y_min - y_range*0.2, y_max + y_range*0.2)

    marker_size = 100 # 统一大小
    line_width = 1.5

    # ====================
    # Step 1: Outlier Detection
    # ====================
    plt.figure(figsize=(7, 5))
    
    # Main Line
    if main_res:
        f = main_res['func']
        p = main_res['params']
        plt.plot(x_line, f(x_line, *p), color=color_line_main, linestyle='--', linewidth=line_width, label='Fitted Line', zorder=5)
        
        # Main Points: 蓝色实心圆，带白色边框，增加分离度
        plt.scatter(main_res['x_fit'], main_res['y_fit'], color=color_main, s=marker_size, 
                    label='Inliers', zorder=10, edgecolors='white', linewidth=1.0, alpha=0.9)
    
    # Outliers: 红色三角形，带白色边框 (用三角形代替叉号，显示为另一种数据)
    if len(outliers_data['x']) > 0:
        plt.scatter(outliers_data['x'], outliers_data['y'], color=color_isomer, s=marker_size, marker='^', 
                    label='Outliers', zorder=10, edgecolors='white', linewidth=1.0, alpha=0.9)

    # Standard Points: 紫色菱形，半透明，大小一致
    if std_data and len(std_data['x']) > 0:
        plt.scatter(std_data['x'], std_data['y'], color=color_std, s=60, marker='D', 
                    label='Standards', zorder=20, edgecolors='k', linewidth=0.8, alpha=0.5)
    
    plt.xlabel('Carbon Number', fontsize=16)
    plt.ylabel('$t_R$ (min)', fontsize=16)
    plt.title(f'{name}', fontsize=14, fontweight='bold')
    plt.xlim(x_min, x_max)
    plt.ylim(y_lim)
    
    # 动态调整Y轴刻度: 最小间隔0.5，但如果刻度太多则增大间隔
    y_range_val = y_lim[1] - y_lim[0]
    if y_range_val <= 4:
        interval = 0.5
    elif y_range_val <= 6:
        interval = 1.0
    elif y_range_val <= 8:
        interval = 1.5
    else:
        interval = 2.0
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(interval))
    
    legend = plt.legend(loc='best', frameon=True, edgecolor='black', fontsize=12, borderpad=0.4)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square")
    
    plt.tight_layout()
    s1_path = os.path.join(output_dir_abs, f'{safe_name}_Step1.png')
    plt.savefig(s1_path)
    plt.close()
    print(f"Step 1 saved to {s1_path}")

    # ====================
    # Step 2: Isomer Fitting
    # ====================
    plt.figure(figsize=(7, 5))
    
    # Main Line
    if main_res:
        f = main_res['func']
        p = main_res['params']
        plt.plot(x_line, f(x_line, *p), color=color_line_main, linestyle='--', linewidth=line_width, label='Main Fit', zorder=5)
        plt.scatter(main_res['x_fit'], main_res['y_fit'], color=color_main, s=marker_size, 
                    label='Main Group', zorder=10, edgecolors='white', linewidth=1.0, alpha=0.9)

    # Isomer Fit
    if isomer_res:
         f = isomer_res['func']
         p = isomer_res['params']
         plt.plot(x_line, f(x_line, *p), color=color_line_isomer, linestyle='--', linewidth=line_width, label='Isomer Fit', zorder=5)
         # 如果是异构体，回归圆形，与Main Group对等，但颜色不同
         plt.scatter(isomer_res['x_fit'], isomer_res['y_fit'], color=color_isomer, s=marker_size, 
                     label='Isomer Group', marker='o', zorder=10, edgecolors='white', linewidth=1.0, alpha=0.9)
    elif len(outliers_data['x']) > 0:
         # Just outliers, no fit found -> 保持三角形
         plt.scatter(outliers_data['x'], outliers_data['y'], color=color_isomer, s=marker_size, 
                     label='Outliers', marker='^', zorder=10, edgecolors='white', linewidth=1.0, alpha=0.9)

    # Standard Points
    if std_data and len(std_data['x']) > 0:
        plt.scatter(std_data['x'], std_data['y'], color=color_std, s=60, marker='D', 
                    label='Standards', zorder=20, edgecolors='k', linewidth=0.8, alpha=0.5)

    plt.xlabel('Carbon Number', fontsize=16)
    plt.ylabel('$t_R$ (min)', fontsize=16)
    plt.title(f'{name} - Step 2: Isomer Fitting', fontsize=14, fontweight='bold')
    plt.xlim(x_min, x_max)
    plt.ylim(y_lim)

    # 动态调整Y轴刻度: 最小间隔0.5，但如果刻度太多则增大间隔
    y_range_val = y_lim[1] - y_lim[0]
    if y_range_val <= 3:
        interval = 0.5
    elif y_range_val <= 6:
        interval = 1.0
    elif y_range_val <= 8:
        interval = 1.5
    else:
        interval = 2.0
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(interval))

    legend = plt.legend(loc='best', frameon=True, edgecolor='black', fontsize=12, borderpad=0.4, ncol=1)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square")
    
    plt.tight_layout()
    s2_path = os.path.join(output_dir_abs, f'{safe_name}_Step2.png')
    plt.savefig(s2_path)
    plt.close()
    print(f"Step 2 saved to {s2_path}")


def main():
    # ==========================================
    # INPUT DATA HERE
    # ==========================================
    # Example: HexCer(d18:0/x:0)
    lipid_name = "SM(d20:1/x:1)"
    
    # 替换成你的数据
    # Carbon Number
    x_data = [11,
16,
19,
23, 
24,







]
    
    # Retention Time
    # 模拟包含异构体的数据
    # 主群: y = 0.4x - 6.3
    # 异构体(34, 36, 38): y = 0.42x - 7.8
    y_data = [4.9,
7.214,
8.458,
10.05,
10.467,







]
    
    # 标准品数据 (可选)
    x_std = []
    y_std = [
]
    # 示例:
    # x_std = [20, 24]
    # y_std = [9.0, 10.7]

    # ==========================================
    # 自动处理
    # ==========================================
    print(f"Processing {lipid_name}...")
    main_res, outliers, isomer_res = analyze_lipid_series(x_data, y_data)
    
    plot_two_steps(lipid_name, x_data, y_data, main_res, outliers, isomer_res, std_data={'x': x_std, 'y': y_std})
    
    print("Done.")

if __name__ == "__main__":
    main()
