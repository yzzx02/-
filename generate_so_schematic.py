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
    color_line_main = '#7f7f7f' # 灰 (Main Fit)
    
    # ==========================================
    # 数据生成
    # ==========================================
    # So(dx:1) 数据
    # x: Carbon Number
    # y: RT
    x_in = np.array([16, 17, 18, 20])
    y_in = np.array([3.562, 3.796, 4.153, 4.463])
    
    # 拟合
    z_in = np.polyfit(x_in, y_in, 1)
    p_in = np.poly1d(z_in)
    
    # ==========================================
    # 绘图
    # ==========================================
    plt.figure(figsize=(7, 5))
    
    # 绘制拟合线
    # 稍微扩展一点范围以便线好看
    x_line = np.linspace(15, 21, 100)
    plt.plot(x_line, p_in(x_line), color=color_line_main, linestyle='--', linewidth=2.5, label='Fitted Line', zorder=5)
    
    # 绘制 Inliers
    plt.scatter(x_in, y_in, color=color_main, s=100, label='Inliers (Used for Fitting)', 
                zorder=10, edgecolors='none')
    
    # 设置标签
    plt.xlabel('Carbon Number', fontsize=16, fontweight='normal')
    plt.ylabel('$t_R$ (min)', fontsize=16, fontweight='normal')
    plt.title('So(dx:1)', fontsize=18, fontweight='bold')
    
    # 调整坐标轴范围
    plt.xlim(15, 21)
    # y轴范围根据数据自动调整稍微多一点
    y_min, y_max = np.min(y_in), np.max(y_in)
    padding = (y_max - y_min) * 0.2
    plt.ylim(y_min - padding, y_max + padding)
    
    # 图例
    legend = plt.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=12, borderpad=0.4)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square") 
    
    plt.tight_layout()
    save_path1 = os.path.join(output_dir, 'So_dx_1_Schematic.png')
    plt.savefig(save_path1, dpi=300)
    print(f"图已生成: {save_path1}")
    plt.close()

if __name__ == "__main__":
    main()
