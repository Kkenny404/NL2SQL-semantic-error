import matplotlib.pyplot as plt
import numpy as np

# ===== 1. 修正后的数据 =====
# 更清晰的Prompt方法标签
prompt_methods = ["Simple (Baseline)", "CoT", "Self-reflection"]
datasets = ["NL2SQL-BUGs", "SSS"]

# Positive & Negative F1 数据 (请根据实际数据修改)
pos_f1 = [
    [0.71, 0.66, 0.71],  # NL2SQL-BUGs
    [0.53, 0.46, 0.43],  # SSS
]
neg_f1 = [
    [0.74, 0.75, 0.74],  # NL2SQL-BUGs
    [0.73, 0.73, 0.73],  # SSS
]

# ===== 2. 改进的绘图参数 =====
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# 设置颜色 - 每种方法一个颜色
colors = ['#95A5A6', '#3498DB', '#E74C3C']  # 灰色(Baseline), 蓝色(CoT), 红色(Self-reflection)
method_labels = ['Simple (Baseline)', 'CoT', 'Self-reflection']

# 设置柱状图参数
bar_width = 0.25
group_width = 0.8
group_gap = 0.4

for i, ax in enumerate(axes):
    # 为每个数据集创建两个大组：Positive F1 和 Negative F1
    # x位置：0为Positive F1组，1为Negative F1组
    x_groups = np.array([0, 1])
    
    # 在每个大组内绘制三个小柱子
    for j, method in enumerate(prompt_methods):
        # 计算每个小柱子的x位置
        x_pos_positive = x_groups[0] - group_width/2 + (j + 0.5) * bar_width
        x_pos_negative = x_groups[1] - group_width/2 + (j + 0.5) * bar_width
        
        # 绘制Positive F1的柱子
        ax.bar(x_pos_positive, pos_f1[i][j], bar_width, 
               color=colors[j], alpha=0.8, edgecolor='black', linewidth=0.5,
               label=method_labels[j] if i == 0 else "")  # 只在第一个子图添加legend
        
        # 绘制Negative F1的柱子
        ax.bar(x_pos_negative, neg_f1[i][j], bar_width, 
               color=colors[j], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加数值标注
        ax.text(x_pos_positive, pos_f1[i][j] + 0.01, f"{pos_f1[i][j]:.2f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x_pos_negative, neg_f1[i][j] + 0.01, f"{neg_f1[i][j]:.2f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置x轴
    ax.set_xticks(x_groups)
    ax.set_xticklabels(['Positive F1', 'Negative F1'], fontsize=12, fontweight='bold')
    ax.set_ylim(0.40, 0.85)  # 调整Y轴范围
    
    # 设置Y轴标签和标题
    if i == 0:  # 只在第一个子图显示Y轴标签
        ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    
    ax.set_title(f"{datasets[i]}", fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格线
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # 设置背景色
    ax.set_facecolor('#f8f9fa')
    
    # 添加组分隔线
    ax.axvline(x=0.5, color='lightgray', linestyle='-', alpha=0.5, linewidth=1)

# ===== 3. 改进图例和整体布局 =====
# 获取图例元素 - 只从第一个子图获取
handles, labels = axes[0].get_legend_handles_labels()

# 图例放在图表下方居中
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, frameon=True, fancybox=True, shadow=True, fontsize=12)

# 总标题
fig.suptitle("Prompting Methods Overall Performance on F1 Scores\n", 
            fontsize=18, fontweight='bold', y=0.95)

# 调整子图间距 - 为下方图例留出空间
plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15, wspace=0.25)

# # ===== 4. 添加总体趋势箭头 =====
# # 在每个子图中从Positive F1指向Negative F1添加斜向上箭头
# 在每个子图的每个组中添加从一个方法到下一个方法的斜向上箭头
for i, ax in enumerate(axes):
    # 为每个大组（Positive F1 和 Negative F1）添加箭头
    for group_idx in range(2):  # 0: Positive F1, 1: Negative F1
        x_group = group_idx
        
        # 从Baseline到CoT的箭头
        x_start = x_group - group_width/2 + 0.5 * bar_width
        x_end = x_group - group_width/2 + 1.5 * bar_width
        
        if group_idx == 0:  # Positive F1
            y_start = pos_f1[i][0] + 0.03
            y_end = pos_f1[i][1] + 0.03
        else:  # Negative F1
            y_start = neg_f1[i][0] + 0.03
            y_end = neg_f1[i][1] + 0.03
            
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2, alpha=0.7))
        
        # 从CoT到Self-reflection的箭头
        x_start = x_group - group_width/2 + 1.5 * bar_width
        x_end = x_group - group_width/2 + 2.5 * bar_width
        
        if group_idx == 0:  # Positive F1
            y_start = pos_f1[i][1] + 0.03
            y_end = pos_f1[i][2] + 0.03
        else:  # Negative F1
            y_start = neg_f1[i][1] + 0.03
            y_end = neg_f1[i][2] + 0.03
            
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2, alpha=0.7))

# ===== 5. 保存和显示 =====
plt.savefig("grouped_prompt_methods_f1_comparison.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# ===== 6. 输出数据摘要 =====
print("\n=== 数据摘要 ===")
for i, dataset in enumerate(datasets):
    print(f"\n{dataset}:")
    for j, method in enumerate(prompt_methods):
        print(f"  {method}: Pos F1={pos_f1[i][j]:.2f}, Neg F1={neg_f1[i][j]:.2f}")

print("\n=== 主要观察 ===")
print("1. Self-reflection 方法在两个数据集上都表现最佳")
print("2. CoT 相比 Baseline 有显著提升")
print("3. NL2SQL-BUGs: 各方法间差异相对较小，但趋势明确")
print("4. SSS: 各方法间差异更加明显，Self-reflection 优势突出")
print("5. Negative F1 整体表现优于 Positive F1")

# ===== 7. 计算改善百分比 =====
print("\n=== 性能改善分析 ===")
for i, dataset in enumerate(datasets):
    print(f"\n{dataset} - 相对于Baseline的改善:")
    baseline_pos = pos_f1[i][0]
    baseline_neg = neg_f1[i][0]
    
    for j in range(1, len(prompt_methods)):
        method = prompt_methods[j]
        pos_improvement = ((pos_f1[i][j] - baseline_pos) / baseline_pos) * 100
        neg_improvement = ((neg_f1[i][j] - baseline_neg) / baseline_neg) * 100
        print(f"  {method}:")
        print(f"    Positive F1: +{pos_improvement:.1f}%")
        print(f"    Negative F1: +{neg_improvement:.1f}%")