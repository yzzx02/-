import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import RANSACRegressor

def main():
    output_dir = '示意图生成'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # ==========================================
    # 样式设置 (保持与 Cer/SM 图一致)
    # ==========================================
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'out' 
    plt.rcParams['ytick.direction'] = 'out'
    
    plt.figure(figsize=(7, 5)) 
    
    # 颜色定义
    color_inlier = '#d62728'  # 红 (Inliers)
    color_outlier = '#2ca02c' # 绿 (Outliers)
    color_line = '#1f77b4'    # 蓝 (Line)
    
    # ==========================================
    # 数据生成
    # ==========================================
    np.random.seed(42)
    
    # X轴: Carbon number of FA (14 - 34)
    # 生成一系列偶数碳数
    x_all = np.arange(14, 35, 2)
    
    # 真实的线性关系 (Inliers)
    # y = 0.45x + 1.2 (参考图)
    slope = 0.45
    intercept = 1.2
    
    y_true = slope * x_all + intercept
    
    # 添加微小噪声作为 Inliers
    y_inliers = y_true + np.random.normal(0, 0.05, len(x_all))
    
    # 构造 Outliers
    # 随机选择几个点变成 Outliers (模拟异构体或噪声)
    # 让 Outliers 偏离直线 (例如向下偏移 1.5 min，模拟另一个异构体，但被 RANSAC 排除)
    n_outliers = 4
    outlier_indices = np.random.choice(len(x_all), n_outliers, replace=False)
    
    # 创建最终的绘图数据
    x_plot = x_all.copy()
    y_plot = y_inliers.copy()
    
    # 修改 Outliers 的 Y 值
    # 偏移量设为 -1.5 到 -2.0 之间
    y_plot[outlier_indices] = y_plot[outlier_indices] - np.random.uniform(1.5, 2.0, n_outliers)
    
    # 还有可能有一个离群点在很远的地方 (参考图右下角)
    # 但为了美观，我们只做这种"疑似异构体但被排除"的点
    
    # 分离 Inliers 和 Outliers 用于绘图
    # 注意：这里我们手动指定谁是 Inlier 谁是 Outlier 来模拟 RANSAC 的结果
    mask_outlier = np.zeros(len(x_all), dtype=bool)
    mask_outlier[outlier_indices] = True
    mask_inlier = ~mask_outlier
    
    x_in = x_plot[mask_inlier]
    y_in = y_plot[mask_inlier]
    
    x_out = x_plot[mask_outlier]
    y_out = y_plot[mask_outlier]
    
    # ==========================================
    # 绘图
    # ==========================================
    
    # 1. 绘制拟合线 (覆盖全范围)
    x_line = np.linspace(13, 35, 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color=color_line, linewidth=2.5, label='Fitted Line', zorder=5)
    
    # 2. 绘制 Inliers (红色圆点)
    plt.scatter(x_in, y_in, color=color_inlier, s=100, label='Inliers (Linear Fitting Points)', 
                zorder=10, edgecolors='black', linewidth=1.0)
    
    # 3. 绘制 Outliers (绿色叉号)
    plt.scatter(x_out, y_out, color=color_outlier, s=100, marker='x', label='Outliers (Not Used for Fitting)', 
                zorder=10, linewidth=2.5)
    
    # 4. 添加公式文本
    # y = 0.45x + 1.2
    # R2 = 0.99...
    text_str = f'$y = {slope:.4f}x + {intercept:.4f}$\n$R^2 = 0.9996$'
    plt.text(15, 14, text_str, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    # 设置标签和标题
    plt.xlabel('Carbon number of FA', fontsize=16, fontweight='normal')
    plt.ylabel('Retention Time (min)', fontsize=16, fontweight='normal')
    plt.title('Cer (d18:0/x:0) RANSAC regression', fontsize=16, fontweight='bold')
    
    # 设置范围
    plt.xlim(13, 36)
    plt.ylim(6, 18)
    
    # 图例
    legend = plt.legend(loc='upper left', 
               frameon=True, 
               edgecolor='black',
               fontsize=11,
               borderpad=0.4,
               handlelength=1.5)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square") 

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'RANSAC_Schematic.png')
    plt.savefig(save_path, dpi=300)
    print(f"示意图已生成: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
