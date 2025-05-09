import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置（可选）
plt.rcParams.update({'font.size': 12})

# 读取数据
df = pd.read_csv('bayesian_optimization_results.csv')

# 计算累积最大值
df['cum_max'] = df['val_r2'].cummax()

# 创建画布（放大画布尺寸）
plt.figure(figsize=(16, 8))  # 宽度和高度均放大

# 使用渐变色
gradient = ['#4B0082', '#9400D3', '#9932CC', '#BA55D3']  # 紫色渐变

# 绘图代码
plt.plot(df['Running_trial'], df['cum_max'],
         color=gradient[2],  # 使用中间紫色
         marker='s',
         markerfacecolor=gradient[3],  # 浅紫色填充
         markeredgecolor='#FFD700',    # 金色边线
         markersize=10,
         linewidth=2,
         label='Max R² Progress')

# 添加突破点标记（增大标记尺寸）
break_points = df[df['val_r2'] == df['cum_max']]
# 突破点标记（金色边框）
plt.scatter(break_points['Running_trial'], 
            break_points['cum_max'],
            edgecolors='#FFD700',
            facecolors='white',
            s=300,
            zorder=4,
            label='New Max')

# 自定义纵坐标范围和刻度（优化显示密度）
plt.ylim(0.972, 0.979)
plt.yticks(np.arange(0.972, 0.979, 0.001))

# 统一放大所有字体（关键修改）
plt.xticks(fontsize=18)    # 坐标轴刻度字体
plt.yticks(fontsize=18)    # 坐标轴刻度字体
plt.xlabel('Running Trial', fontsize=20)  # 坐标轴标签
plt.ylabel('Validation R²', fontsize=20)
plt.title('Validation R² Cumulative Maximum Progress', 
          fontsize=22, pad=25)  # 标题字体

# 添加数据标签（增大字体）
for x, y in zip(break_points['Running_trial'], break_points['cum_max']):
    plt.annotate(f'{y:.3f}', (x, y), 
                 textcoords="offset points", 
                 xytext=(0,15),  # 增大标签偏移量
                 ha='center', 
                 fontsize=12,  # 增大标签字体
                 weight='bold')

# 添加网格和辅助线（增大线宽）
plt.grid(True, linestyle='--', alpha=0.8, linewidth=0.8)
plt.axhline(y=df['cum_max'].max(), 
            color='#3A8FEA', 
            linestyle=':', 
            linewidth=2,  # 增大线宽
            label=f'Final Max: {df["cum_max"].max():.3f}')

# 调整图例（增大字体和边框）
plt.legend(fontsize=18, loc='lower right', 
           frameon=True,  # 显示图例边框
           shadow=True,   # 添加阴影
           facecolor='white')

# 调整整体边距
plt.tight_layout(pad=3)

# 显示图表
plt.show()
