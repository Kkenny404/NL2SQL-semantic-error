import matplotlib.pyplot as plt
import numpy as np

# ===== 1. 数据设置 =====
datasets = ["NL2SQL-BUGs", "SSS"]

# Without Parser vs With Parser 的 F1 数据
# 请替换为你的实际数据
without_parser_pos_f1 = [0.71, 0.53]  # 示例数据，请替换
without_parser_neg_f1 = [0.74, 0.73]  # SSS用你刚计算的0.72

with_parser_pos_f1 = [0.71, 0.58]     # 示例数据，请替换为实际值
with_parser_neg_f1 = [0.74, 0.72]     # 示例数据，请替换为实际值

# ===== 2. 创建图表 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# 设置颜色 - 按数据集分配颜色
color_nl2sql = '#2E86AB'     # 蓝色 - NL2SQL-BUGs
color_sss = '#E74C3C'        # 红色 - SSS  
color_baseline = '#95A5A6'   # 浅灰色 - Without Parser (基线)

bar_width = 0.35
x = np.arange(len(datasets))

# 为每个数据集分配颜色
dataset_colors_with_parser = [color_nl2sql, color_sss]

for i, ax in enumerate(axes):
    if i == 0:  # Positive F1
        # Without Parser (统一灰色)
        bars1 = ax.bar(x - bar_width/2, without_parser_pos_f1, bar_width, 
                      label="Without Parser", color=color_baseline, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # With Parser (按数据集分色)
        bars2 = []
        for j in range(len(datasets)):
            bar = ax.bar(x[j] + bar_width/2, with_parser_pos_f1[j], bar_width, 
                        color=dataset_colors_with_parser[j], 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2.append(bar)
        
        # 添加数值标注
        for j, (val1, val2) in enumerate(zip(without_parser_pos_f1, with_parser_pos_f1)):
            ax.text(x[j] - bar_width/2, val1 + 0.01, f"{val1:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(x[j] + bar_width/2, val2 + 0.01, f"{val2:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title("Positive F1 Score", fontsize=14, fontweight='bold', pad=15)
        
    else:  # Negative F1
        # Without Parser (统一灰色)
        bars1 = ax.bar(x - bar_width/2, without_parser_neg_f1, bar_width, 
                      label="Without Parser", color=color_baseline, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # With Parser (按数据集分色)
        bars2 = []
        for j in range(len(datasets)):
            bar = ax.bar(x[j] + bar_width/2, with_parser_neg_f1[j], bar_width, 
                        color=dataset_colors_with_parser[j], 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2.append(bar)
        
        # 添加数值标注
        for j, (val1, val2) in enumerate(zip(without_parser_neg_f1, with_parser_neg_f1)):
            ax.text(x[j] - bar_width/2, val1 + 0.01, f"{val1:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(x[j] + bar_width/2, val2 + 0.01, f"{val2:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title("Negative F1 Score", fontsize=14, fontweight='bold', pad=15)

    # 设置坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=13)
    ax.set_ylim(0.45, 0.85)
    
    # Y轴标签
    if i == 0:
        ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    
    # 网格和背景
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')

# ===== 3. 添加改进指示 =====
# 为显著改进添加箭头和百分比
def add_improvement_arrow(ax, x_pos, before_val, after_val, dataset_name, dataset_color):
    """添加改进指示箭头"""
    improvement = ((after_val - before_val) / before_val) * 100
    
    if improvement > 2:  # 只在显著改进时添加箭头
        # 使用对应数据集的颜色，但稍微调深一些
        arrow_color = dataset_color
        
        # 计算箭头位置
        arrow_y = max(before_val, after_val) + 0.05
        
        # 添加箭头
        ax.annotate('', xy=(x_pos + bar_width/2, after_val + 0.05), 
                   xytext=(x_pos - bar_width/2, before_val + 0.05),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
        
        # 添加改进百分比
        ax.text(x_pos, arrow_y, f"+{improvement:.1f}%", 
               ha='center', va='bottom', fontsize=10, 
               color=arrow_color, fontweight='bold')

# 检查并添加改进箭头
for j in range(len(datasets)):
    # Positive F1改进
    add_improvement_arrow(axes[0], x[j], without_parser_pos_f1[j], 
                         with_parser_pos_f1[j], datasets[j], dataset_colors_with_parser[j])
    
    # Negative F1改进
    add_improvement_arrow(axes[1], x[j], without_parser_neg_f1[j], 
                         with_parser_neg_f1[j], datasets[j], dataset_colors_with_parser[j])

# ===== 4. 图例和标题 =====
# 创建完整的图例
handles = [
    plt.Rectangle((0,0),1,1, color=color_baseline, alpha=0.8, label='Original SQL Only (Baseline)'),
    plt.Rectangle((0,0),1,1, color=color_nl2sql, alpha=0.8, label='NL2SQL-BUGs (With Parser)'),
    plt.Rectangle((0,0),1,1, color=color_sss, alpha=0.8, label='SSS (With Parser)')
]

# 图例放在底部
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, frameon=True, fancybox=True, shadow=True, fontsize=12)

# 总标题
fig.suptitle("Input Adjustment on the SQL Query -- Overall Performance \n(Original SQL Only vs With Extra Parsed SQL)", 
            fontsize=16, fontweight='bold', y=0.92)

# ===== 5. 布局调整 =====
plt.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.18, wspace=0.25)

# ===== 6. 保存和显示 =====
plt.savefig("parser_before_after_comparison_fixed.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# ===== 7. 数据摘要和使用说明 =====
print("=== Parser效果对比数据 ===")
print("请将以下示例数据替换为你的实际数据:\n")

print("Without Parser:")
for i, dataset in enumerate(datasets):
    print(f"  {dataset}: Pos F1={without_parser_pos_f1[i]:.2f}, Neg F1={without_parser_neg_f1[i]:.2f}")

print("\nWith Parser:")
for i, dataset in enumerate(datasets):
    print(f"  {dataset}: Pos F1={with_parser_pos_f1[i]:.2f}, Neg F1={with_parser_neg_f1[i]:.2f}")

print("\n=== 使用说明 ===")
print("📊 图表特色:")
print("  - 灰色柱子: Without Parser (基线)")
print("  - 蓝色柱子: NL2SQL-BUGs数据集 (With Parser)")
print("  - 红色柱子: SSS数据集 (With Parser)")
print("  - 彩色箭头和百分比: 对应数据集颜色的改进指示")

print("\n🔧 需要更新的数据:")
print("  1. without_parser_pos_f1 = [NL2SQL值, SSS值]")
print("  2. without_parser_neg_f1 = [NL2SQL值, 0.72]  # SSS已计算")
print("  3. with_parser_pos_f1 = [你的实际数据]")
print("  4. with_parser_neg_f1 = [你的实际数据]")

print("\n💡 图表亮点:")
print("  - 清晰的Before/After对比")
print("  - 数据集一致的颜色编码")
print("  - 自动计算和显示改进百分比")
print("  - 专业的学术图表格式")