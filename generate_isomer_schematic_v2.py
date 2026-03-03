import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    output_dir = '示意图生成'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # ==========================================
    # 样式设置
    # ==========================================
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'out' 
    plt.rcParams['ytick.direction'] = 'out'
    
    # 颜色定义
    color_main = '#1f77b4'    # 蓝 (Main Group)
    color_isomer = '#d62728'  # 红 (Isomer Group)
    color_line_main = '#7f7f7f' # 灰 (Main Fit)
    color_line_isomer = '#d62728' # 红 (Isomer Fit)
    
    # ==========================================
    # 数据生成
    # ==========================================
    # 范围: 32 - 42 C
    # 主趋势: y = 0.4x - 6.3
    slope = 0.4
    intercept = -6.3
    
    # 异构体趋势: 斜率近似，截距不同 (向下偏移 1.5 min)
    slope_iso = 0.42 # 稍微有点差别
    intercept_iso = -7.8
    
    # 定义 X 轴点
    # Inliers (主群): 大部分点
    x_in = np.array([32, 33, 35, 37, 39, 41, 42])
    
    # Outliers (异构体群): 4个点，在同一侧
    x_out = np.array([34, 36, 38, 40])
    
    # 生成 Y 值 (带噪声)
    np.random.seed(42)
    y_in = slope * x_in + intercept + np.random.normal(0, 0.05, len(x_in))
    y_out = slope_iso * x_out + intercept_iso + np.random.normal(0, 0.05, len(x_out))
    
    # 拟合
    z_in = np.polyfit(x_in, y_in, 1)
    p_in = np.poly1d(z_in)
    
    z_out = np.polyfit(x_out, y_out, 1)
    p_out = np.poly1d(z_out)
    
    # ==========================================
    # 图 1: 离群点检测 (Step 1)
    # ==========================================
    plt.figure(figsize=(7, 5))
    
    # 绘制拟合线 (主)
    x_line = np.linspace(31.5, 42.5, 100)
    plt.plot(x_line, p_in(x_line), color=color_line_main, linestyle='--', linewidth=2.5, label='Fitted Line', zorder=5)
    
    # 绘制 Inliers
    plt.scatter(x_in, y_in, color=color_main, s=100, label='Inliers (Used for Fitting)', 
                zorder=10, edgecolors='none')
    
    # 绘制 Outliers (标记为 Excluded)
    plt.scatter(x_out, y_out, color=color_isomer, s=100, marker='x', linewidth=2.5, 
                label=f'Outliers (Excluded, n={len(x_out)})', zorder=10)
    
    # 设置标签
    plt.xlabel('Carbon Number', fontsize=16, fontweight='normal')
    plt.ylabel('$t_R$ (min)', fontsize=16, fontweight='normal')
    plt.title('HexCer(d18:0/x:0) and isomers', fontsize=18, fontweight='bold')
    
    plt.xlim(31, 43)
    plt.ylim(5.5, 11.5)
    
    # 图例
    legend = plt.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, borderpad=0.4)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square") 
    
    plt.tight_layout()
    save_path1 = os.path.join(output_dir, 'HexCer_Isomer_Step1.png')
    plt.savefig(save_path1, dpi=300)
    print(f"图1已生成: {save_path1}")
    plt.close()
    
    # ==========================================
    # 图 2: 异构体拟合 (Step 2)
    # ==========================================
    plt.figure(figsize=(7, 5))
    
    # 绘制拟合线 (主)
    plt.plot(x_line, p_in(x_line), color=color_line_main, linestyle='--', linewidth=2.5, label='Main Fit', zorder=5)
    
    # 绘制拟合线 (异构体)
    plt.plot(x_line, p_out(x_line), color=color_line_isomer, linestyle='--', linewidth=2.5, label='Isomer Fit', zorder=5)
    
    # 绘制 Inliers
    plt.scatter(x_in, y_in, color=color_main, s=100, label='Main Group', 
                zorder=10, edgecolors='none')
    
    # 绘制 Outliers (现在是 Isomer Group)
    plt.scatter(x_out, y_out, color=color_isomer, s=100, marker='o', label='Isomer Group', 
                zorder=10, edgecolors='none') # 改回圆点，表示它们也是有效数据
    
    # 设置标签
    plt.xlabel('Carbon Number', fontsize=16, fontweight='normal')
    plt.ylabel('$t_R$ (min)', fontsize=16, fontweight='normal')
    plt.title('HexCer(d18:0/x:0) and isomers', fontsize=18, fontweight='bold')
    
    plt.xlim(31, 43)
    plt.ylim(5.5, 11.5)
    
    # 图例
    legend = plt.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, borderpad=0.4, ncol=2)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square") 
    
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, 'HexCer_Isomer_Step2.png')
    plt.savefig(save_path2, dpi=300)
    print(f"图2已生成: {save_path2}")
    plt.close()

if __name__ == "__main__":
    main()
