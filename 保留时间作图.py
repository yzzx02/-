import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
# 创建保存图片的文件夹
if not os.path.exists('拟合图'):
    os.makedirs('拟合图')

# 读取Excel文件
df = pd.read_excel('RT结果.xlsx', sheet_name='Sheet1')

# 筛选有效数据：要求FA_Carbon和RT列都有有效数值
df = df.dropna(subset=['FA_Carbon', 'RT'])

# 创建跳过记录字典
skipped_indices = {
    '数据点不足': [],
    '碳数值相同': [],
    '拟合失败': []
}
# 处理跳过的index计数器
skip_count = 0
# 对每个index分组处理
for index_value, group in df.groupby('index'):
    skip = False
    if len(group) < 2:  # 需要至少2个点才能拟合
        print(f"跳过index {index_value}：数据点不足")
        skipped_indices['数据点不足'].append(index_value)
        skip_count += 1
        skipped = True
        continue
    
    # 检查x值是否完全相同
    if group['FA_Carbon'].nunique() == 1:
        print(f"跳过index {index_value}：所有FA碳数值相同 ({group['FA_Carbon'].iloc[0]})")
        skipped_indices['碳数值相同'].append(index_value)
        skip_count += 1
        skipped = True
        continue
    
    # 获取P列(series1)的值作为标题
    series_name = group['series'].iloc[0] #取series第一个值
    
    # 准备数据
    x = group['FA_Carbon'].astype(float)
    y = np.log10(group['RT'].astype(float))  # 计算log(RT)
    
    try:
        # 尝试进行线性回归拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line = slope * x + intercept
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制数据点和拟合线
        plt.scatter(x, y, s=40, color='blue', label='Data points')
        plt.plot(x, line, 'r-', linewidth=2, label=f'线性拟合: log(RT) = {slope:.4f}·C + {intercept:.4f}\n $R^2$ = {r_value**2:.4f}')
        
        # 添加标签和标题
        plt.xlabel('FA碳数', fontsize=12)
        plt.ylabel('log(保留时间/min)', fontsize=12)
        plt.title(f'{series_name} (Index {index_value})', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 保存图表
        plt.savefig(f'拟合图/拟合图{int(index_value)}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ValueError as e:
        print(f"跳过index {index_value}: 线性回归失败 - {str(e)}")
        skip_count += 1
        skipped = True

print(f"图表生成完成！共处理 {len(df.groupby('index'))} 个索引组")
print(f"跳过的index总数: {skip_count}")
if skipped_indices['数据点不足']:
    print(f"\n因数据点不足跳过({len(skipped_indices['数据点不足'])}个):")
    print(", ".join(map(str, skipped_indices['数据点不足'])))
    
if skipped_indices['碳数值相同']:
    print(f"\n因所有碳数值相同跳过({len(skipped_indices['碳数值相同'])}个):")
    print(", ".join(map(str, skipped_indices['碳数值相同'])))
    
if skipped_indices['拟合失败']:
    print(f"\n因拟合失败跳过({len(skipped_indices['拟合失败'])}个):")
    for idx, reason in skipped_indices['拟合失败']:
        print(f"  index {idx}: {reason}")