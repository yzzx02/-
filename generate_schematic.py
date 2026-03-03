import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    output_dir = '示意图生成'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 设置绘图风格 - 仿照参考图的 clean 风格
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'out' 
    plt.rcParams['ytick.direction'] = 'out'
    
    plt.figure(figsize=(7, 5)) 
    
    # 颜色定义 (参考图: 蓝, 绿, 灰, 橙) + 紫
    colors = [
        '#5B9BD5',  # 1: 蓝色 
        '#70AD47',  # 2: 绿色
        '#A5A5A5',  # 3: 灰色
        '#ED7D31',  # 4: 橙色
        '#7030A0',  # 5: 紫色 
    ]
    
    # 定义每条曲线的 "锚点" (x, y) 用于拟合二次曲线
    anchors = {
        1: [(30, 6.5), (42, 13.5), (56, 17.5)],   
        2: [(32, 6.0), (42, 12.0), (50, 15.5)],   
        3: [(36, 7.5), (46, 12.0), (54, 14.5)],  
        4: [(40, 8.5), (48, 11.5), (54, 13.0)],  
        5: [(44, 9.5), (50, 11.0), (54, 12.0)],  
    }
    
    # 定义每条曲线的 X 轴数据点 (偶数碳数)
    x_points_dict = {
        1: np.concatenate([np.arange(30, 41, 1), np.arange(42, 46, 2), [54]]), # 30-40 逐个点，42-44 偶数，54 一个点
        2: np.arange(32, 52, 2),
        3: np.arange(36, 56, 2),
        4: np.arange(40, 54, 2),
        5: np.array([46, 50, 54]),
    }

    legend_handles = []
    legend_labels = []
    
    # 固定随机种子，保证每次生成的偏移一致
    np.random.seed(42)

    for unsat in [1, 2, 3, 4, 5]:
        # 1. 计算拟合参数
        pts = anchors[unsat]
        x_fit = np.array([p[0] for p in pts])
        y_fit = np.array([p[1] for p in pts])
        z = np.polyfit(x_fit, y_fit, 2) # 二次多项式拟合
        p = np.poly1d(z)
        
        # 2. 生成绘图用的平滑曲线数据 (虚线)
        # 稍微延伸一点点
        x_smooth = np.linspace(x_points_dict[unsat][0] - 1, x_points_dict[unsat][-1] + 1, 100)
        y_smooth = p(x_smooth)
        
        # 3. 生成实际数据点 (圆点)
        x_data = x_points_dict[unsat]
        y_data = p(x_data)
        
        # 添加微小随机偏移 (让数据看起来不那么"假")
        # 偏移量控制在 0.15 左右，不影响整体趋势
        noise = np.random.normal(0, 0.15, size=len(y_data))
        y_data = y_data + noise
        
        color = colors[unsat-1]
        
        # 画虚线 (Thick dotted line)
        plt.plot(x_smooth, y_smooth, color=color, linestyle=':', linewidth=2.5, alpha=0.8)
        
        # 画数据点 (Large solid circles)
        scatter = plt.scatter(x_data, y_data, color=color, s=100, zorder=10, edgecolors='none')
        
        legend_handles.append(scatter)
        legend_labels.append(str(unsat))

    # 设置轴范围
    plt.xlim(28, 60)
    plt.ylim(5, 19)
    
    # 设置标签
    plt.xlabel('Carbon Number', fontsize=16, fontweight='normal')
    plt.ylabel('$t_R$ (min)', fontsize=16, fontweight='normal')
    
    # 标题
    plt.title('Cer(d18:1/x:y)', fontsize=18, fontweight='bold')

    # 图例
    legend = plt.legend(legend_handles, legend_labels, 
               loc='upper left', 
               ncol=5, 
               columnspacing=0.2,
               handletextpad=0.1,
               frameon=True, 
               edgecolor='black',
               fontsize=12,
               borderpad=0.4,
               handlelength=1.0)
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square") 

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'Cer_d18_1_Schematic.png')
    plt.savefig(save_path, dpi=300)
    print(f"示意图已生成: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
