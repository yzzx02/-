import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    output_dir = '示意图生成'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # ==========================================
    # 样式设置 (保持与 Cer 图一致)
    # ==========================================
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.direction'] = 'out' 
    plt.rcParams['ytick.direction'] = 'out'
    
    plt.figure(figsize=(7, 5)) 
    
    # 颜色定义 (Cer 风格)
    colors = [
        '#5B9BD5',  # 1: 蓝色 
        '#70AD47',  # 2: 绿色
        '#A5A5A5',  # 3: 灰色
        '#ED7D31',  # 4: 橙色
        '#7030A0',  # 5: 紫色 
    ]
    
    # ==========================================
    # 数据逻辑 (SM 数据特点)
    # ==========================================
    # 目标: 稍微凹一点点的二次函数 (a > 0)
    # 范围: X ~ 30-45, Y ~ 5.5-11
    
    # 基础参数 (调整以适应 5.5-11 min 范围)
    base_a = 0.002 # 极小的二次项系数
    base_b = 0.15  # 斜率降低
    base_c = 0.2   # 截距调整
    
    # 不同不饱和度的偏移 (RT 随不饱和度增加而降低)
    # 间距缩小以适应更窄的 Y 轴范围
    offsets = [0, -0.25, -0.5, -0.75, -1.0]
    
    # 定义X轴范围 (参考 SM 图片)
    ranges = {
        1: (30, 44),
        2: (32, 44),
        3: (34, 44),
        4: (34, 45),
        5: (34, 44)
    }

    legend_handles = []
    legend_labels = []
    
    np.random.seed(10) # 固定随机种子

    for i, unsat in enumerate([1, 2, 3, 4, 5]):
        x_start, x_end = ranges[unsat]
        
        # 针对 4 和 5 个不饱和度进行特殊处理
        if unsat <= 3:
            # 1-3: 连续的点
            x_data = np.arange(x_start, x_end + 1, 1)
            # 标准参数
            a = base_a
            b = base_b
            c = base_c + offsets[i]
        elif unsat == 4:
            # 4: 稀疏点
            x_data = np.array([34, 36, 37, 39, 40, 42, 44, 45])
            # 参数微调
            a = base_a * 1.1      # 稍微弯一点
            b = base_b * 0.95     # 线性部分斜率稍减
            # 修正截距: 必须确保比 unsat=3 (offset=-0.5) 低
            # offset[4] 是 -0.75.
            # 之前加了0.5导致反超。现在只加一个小补偿或者不加，取决于a/b变化导致的影响
            # x~40时, a变大导致y增大约0.3, b变小导致y减小约0.3, 抵消了。
            # 所以截距保持 offset 应该就行，为了稳妥，再减一点点
            c = base_c + offsets[i] - 0.1 
        elif unsat == 5:
            # 5: 更稀疏点
            x_data = np.array([36, 38, 40, 41, 43])
            # 参数微调
            a = base_a * 0.9      # 稍微平一点
            b = base_b * 0.92     # 斜率更低
            # x~40时, a减小导致y减小0.3, b减小导致y减小0.5. 总共减小0.8.
            # offset[5] 是 -1.0. 
            # 如果不补偿，会掉下去太深(与unsat4间距过大)。
            # 补偿0.6左右，让它只比unsat4低一点
            c = base_c + offsets[i] + 0.6
            
        y_clean = a * x_data**2 + b * x_data + c
        
        # 添加随机噪声 (Cer 风格: 0.15)
        np.random.seed(10 + i) # 不同的随机种子避免噪声模式一样
        noise = np.random.normal(0, 0.15, len(x_data))
        y_data = y_clean + noise
        
        # 绘制拟合曲线 (显示趋势)
        x_line = np.linspace(x_start - 0.5, x_end + 0.5, 100)
        y_line = a * x_line**2 + b * x_line + c
        
        color = colors[i]
        
        plt.plot(x_line, y_line, color=color, linestyle=':', linewidth=2.5, alpha=0.8)
        
        # 绘制散点 (Cer 风格: 大实心圆, 无边框)
        scatter = plt.scatter(x_data, y_data, color=color, s=100, zorder=10, edgecolors='none')
        
        legend_handles.append(scatter)
        legend_labels.append(str(unsat))

    # 设置标签和标题
    plt.xlabel('Carbon Number', fontsize=16, fontweight='normal')
    plt.ylabel('$t_R$ (min)', fontsize=16, fontweight='normal')
    plt.title('SM(d18:1/x:y)', fontsize=18, fontweight='bold')
    
    # 设置范围 (SM 数据范围)
    plt.xlim(29, 46)
    plt.ylim(5, 12)
    plt.xticks([30, 35, 40, 45])
    
    # 图例 (Cer 风格: 左上角, 带边框)
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
    
    save_path = os.path.join(output_dir, 'SM_d18_1_Schematic.png')
    plt.savefig(save_path, dpi=300)
    print(f"示意图已生成: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
