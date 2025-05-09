import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
df_filtered = pd.read_csv('bayesian_optimization_results.csv')

# 设置不同的标记样式和大小
marker_styles = ['o', 's', 'D', '^', 'v', 'p', '*', 'X']
marker_sizes = [60, 80, 100, 120, 140, 160, 180, 200]

# 创建图形
plt.figure(figsize=(8, 5),dpi=300)

# 1. 绘制所有普通散点
for i, wavelet in enumerate(df_filtered['wavelet'].unique()):
    subset = df_filtered[df_filtered['wavelet'] == wavelet]
    plt.scatter(
        subset['level'], 
        subset['val_r2'],
        s=marker_sizes[i % len(marker_sizes)],
        marker=marker_styles[i % len(marker_styles)],
        label=wavelet,
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5
    )

# 2. 添加最佳结果标记
best_point = df_filtered.loc[df_filtered['val_r2'].idxmax()]
plt.scatter(
    best_point['level'],
    best_point['val_r2'],
    s=250,
    marker='*',
    color='gold',
    edgecolors='k',
    label='Best Result',
    zorder=10
)

# 3. 在右下角添加最佳参数文本
best_params_text = (
    f"Best Parameters:\n"
    f"• R²: {best_point['val_r2']:.4f}\n"
    f"• Wavelet: {best_point['wavelet']}\n"
    f"• Level: {int(best_point['level'])}\n"
    f"• Heads: {int(best_point['num_heads'])}\n"
    f"• Units: {int(best_point['transformer_units'])}\n"
    f"• Dense: {int(best_point['dense_units'])}\n"
    f"• Dropout: {best_point['dropout_rate']:.3f}\n"
    f"• LR: {best_point['learning_rate']:.6f}"
)

plt.text(
    1.07, 0.01,  # 右下角坐标 (x,y ∈ [0,1])
    best_params_text,
    transform=plt.gca().transAxes,  # 使用相对坐标
    fontsize=10,
    verticalalignment='bottom',
    horizontalalignment='left',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.4,
        edgecolor='gray',
        pad=0.5
    )
)

# 4. 添加其他图形元素
# plt.title("Validation R² Value by Wavelet Type and Decomposition Level", fontsize=12, pad=20)
plt.xlabel("Decomposition Level", fontsize=12)
plt.ylabel("Validation R² Value", fontsize=12)

# 优化图例位置
legend = plt.legend(
    title="Wavelet Function",
    title_fontsize=12,
    fontsize=12,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.
)

# 调整坐标轴范围
plt.xlim(df_filtered['level'].min() - 0.5, df_filtered['level'].max() + 0.5)
plt.ylim(df_filtered['val_r2'].min() - 0.002, df_filtered['val_r2'].max() + 0.002)

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
