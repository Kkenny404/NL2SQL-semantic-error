import matplotlib.pyplot as plt
import numpy as np

# ===== 1. 修正后的数据 =====
# 更清晰的Schema类型标签
schema_types = ["Baseline\n(Basic Schema)", "No Schema", "Schema with\nKey Information"]
datasets = ["NL2SQL-BUGs", "SSS"]

# Positive & Negative F1 数据
pos_f1 = [
    [0.71, 0.74, 0.70],  # NL2SQL-BUGs
    [0.53, 0.60, 0],  # SSS (第三列数据缺失)
]
neg_f1 = [
    [0.74, 0.72, 0.74],  # NL2SQL-BUGs
    [0.73, 0.72, 0],  # SSS (第三列数据缺失)
]

# ===== 2. 改进的绘图参数 =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# 设置颜色 - 更容易区分
colors_pos = '#2E86AB'  # 深蓝色
colors_neg = '#F24236'  # 红色

bar_width = 0.35
x = np.arange(len(schema_types))

for i, ax in enumerate(axes):
    # 绘制柱状图 - 处理缺失数据
    pos_data = [val for val in pos_f1[i] if val is not None]
    neg_data = [val for val in neg_f1[i] if val is not None]
    x_indices = [j for j, val in enumerate(pos_f1[i]) if val is not None]
    
    bars_pos = ax.bar(np.array(x_indices) - bar_width/2, pos_data, bar_width, 
                      label="Positive F1", color=colors_pos, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_neg = ax.bar(np.array(x_indices) + bar_width/2, neg_data, bar_width, 
                      label="Negative F1", color=colors_neg, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)

    # 添加数值标注 - 只对有数据的列
    for j, (pos_val, neg_val) in enumerate(zip(pos_f1[i], neg_f1[i])):
        if pos_val is not None and neg_val is not None:
            # Positive F1 标注
            ax.text(x[j] - bar_width/2, pos_val + 0.01, f"{pos_val:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Negative F1 标注
            ax.text(x[j] + bar_width/2, neg_val + 0.01, f"{neg_val:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(schema_types, fontsize=10, ha='center')
    ax.set_ylim(0.45, 0.80)  # 调整Y轴范围更合理
    
    # 设置Y轴标签和标题
    if i == 0:  # 只在第一个子图显示Y轴标签
        ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    
    ax.set_title(f"{datasets[i]}", fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # 设置背景色
    ax.set_facecolor('#f8f9fa')

# ===== 3. 改进图例和整体布局 =====
# 获取图例元素
handles, labels = axes[0].get_legend_handles_labels()

# 图例放在图表下方居中 - 调整到合适位置
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, frameon=True, fancybox=True, shadow=True, fontsize=12)

# 总标题 - 更简洁
fig.suptitle("Impact of Schema Information on F1 Scores\n(GPT-4.1 Performance)", 
            fontsize=16, fontweight='bold', y=0.95)

# 调整子图间距 - 为下方图例留出空间
plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15, wspace=0.25)

# ===== 4. 添加性能分析注释 =====
# 在NL2SQL-BUGs子图添加观察注释 - 指向No Schema列
axes[0].annotate('Overall best\nperformance', 
                xy=(1, 0.74), xytext=(1.3, 0.77),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=10, ha='center', color='#333333')

# 在SSS子图添加观察注释 - 指向第三列缺失位置
axes[1].annotate('Spider database\nincomplete on key information', 
                xy=(2, 0.50), xytext=(1.9, 0.56),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=10, ha='center', color='#333333')

axes[1].annotate('Better performance', 
                xy=(0.85, 0.7), xytext=(1, 0.77),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=10, ha='center', color='#333333')

# ===== 5. 保存和显示 =====
plt.savefig("improved_schema_f1_comparison.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# ===== 6. 输出数据摘要 =====
print("\n=== 数据摘要 ===")
for i, dataset in enumerate(datasets):
    print(f"\n{dataset}:")
    for j, schema in enumerate(schema_types):
        schema_clean = schema.replace('\n', ' ')
        print(f"  {schema_clean}: Pos F1={pos_f1[i][j]:.2f}, Neg F1={neg_f1[i][j]:.2f}")

print("\n=== 主要观察 ===")
print("1. NL2SQL-BUGs: 'No Schema' 条件下整体表现最佳")
print("2. SSS: 'Schema with Key Information' 数据不完整，无法比较")
print("3. SSS数据集中，'No Schema' 相比 'Baseline' 显著提升Positive F1 (0.53→0.60)")
print("4. Negative F1在NL2SQL-BUGs数据集上相对稳定")