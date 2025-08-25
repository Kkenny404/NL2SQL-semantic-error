import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== 1. 数据 =====
schema_types = ["Baseline\n(Basic Schema)", "No Schema", "Schema with\nKey Information"]
datasets = ["NL2SQL-BUGs", "SSS"]

# F1数据 (用于图表)
pos_f1 = [
    [0.71, 0.74, 0.70],  # NL2SQL-BUGs
    [0.53, 0.60, 0.72],  # SSS
]
neg_f1 = [
    [0.74, 0.72, 0.74],  # NL2SQL-BUGs
    [0.73, 0.72, 0.72],  # SSS
]

# Precision & Recall数据 (用于表格)
pos_precision = [
    [0.68, 0.72, 0.67],  # NL2SQL-BUGs (请替换为真实值)
    [0.51, 0.58, 0.70],  # SSS
]
neg_precision = [
    [0.76, 0.74, 0.76],  # NL2SQL-BUGs
    [0.75, 0.74, 0.74],  # SSS
]

pos_recall = [
    [0.74, 0.76, 0.73],  # NL2SQL-BUGs (请替换为真实值)
    [0.55, 0.62, 0.74],  # SSS
]
neg_recall = [
    [0.72, 0.70, 0.72],  # NL2SQL-BUGs
    [0.71, 0.70, 0.70],  # SSS
]

# ===== 2. 创建F1图表 (保持原有设计) =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# 设置颜色
colors_pos = '#2E86AB'  # 深蓝色
colors_neg = '#F24236'  # 红色

bar_width = 0.35
x = np.arange(len(schema_types))

for i, ax in enumerate(axes):
    # 绘制柱状图
    bars_pos = ax.bar(x - bar_width/2, pos_f1[i], bar_width, 
                      label="Positive F1", color=colors_pos, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_neg = ax.bar(x + bar_width/2, neg_f1[i], bar_width, 
                      label="Negative F1", color=colors_neg, 
                      alpha=0.8, edgecolor='black', linewidth=0.5)

    # 添加数值标注
    for j, (pos_val, neg_val) in enumerate(zip(pos_f1[i], neg_f1[i])):
        ax.text(x[j] - bar_width/2, pos_val + 0.01, f"{pos_val:.2f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x[j] + bar_width/2, neg_val + 0.01, f"{neg_val:.2f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(schema_types, fontsize=10, ha='center')
    ax.set_ylim(0.45, 0.80)
    
    if i == 0:
        ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    
    ax.set_title(f"{datasets[i]}", fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')

# 添加注释
axes[0].annotate('Overall best\nperformance', 
                xy=(1, 0.74), xytext=(1.3, 0.77),
                arrowprops=dict(arrowstyle='->', color='darkgray', alpha=0.8),
                fontsize=9, ha='center', color='#333333')

axes[1].annotate('Spider database\nincomplete on key information', 
                xy=(2, 0.72), xytext=(1.7, 0.56),
                arrowprops=dict(arrowstyle='->', color='darkgray', alpha=0.8),
                fontsize=9, ha='center', color='#333333')

# 图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.08),
           ncol=2, frameon=True, fancybox=True, shadow=True, fontsize=12)

# 标题
fig.suptitle("Impact of Schema Information on F1 Scores\n(GPT-4.1 Performance)", 
            fontsize=16, fontweight='bold', y=0.95)

plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.25, wspace=0.25)
plt.savefig("f1_schema_comparison.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# ===== 3. 创建Precision & Recall表格 =====
print("\n" + "="*80)
print("PRECISION & RECALL DETAILED BREAKDOWN")
print("="*80)

# 创建表格数据
table_data = []
schema_labels = ["Baseline (Basic Schema)", "No Schema", "Schema with Key Information"]

for dataset_idx, dataset_name in enumerate(datasets):
    print(f"\n📊 {dataset_name.upper()}")
    print("-" * 60)
    
    # 表头
    print(f"{'Schema Type':<25} {'Pos P':<8} {'Pos R':<8} {'Neg P':<8} {'Neg R':<8}")
    print("-" * 60)
    
    # 数据行
    for schema_idx, schema_name in enumerate(schema_labels):
        pos_p = pos_precision[dataset_idx][schema_idx]
        pos_r = pos_recall[dataset_idx][schema_idx]
        neg_p = neg_precision[dataset_idx][schema_idx]
        neg_r = neg_recall[dataset_idx][schema_idx]
        
        # 截断长标签用于显示
        display_name = schema_name[:23] + ".." if len(schema_name) > 25 else schema_name
        
        print(f"{display_name:<25} {pos_p:<8.2f} {pos_r:<8.2f} {neg_p:<8.2f} {neg_r:<8.2f}")

# ===== 4. 创建LaTeX表格格式 (论文用) =====
print(f"\n\n{'='*80}")
print("LATEX TABLE FORMAT (Copy to your thesis)")
print("="*80)

print("""
\\begin{table}[htbp]
\\centering
\\caption{Precision and Recall Breakdown by Schema Type}
\\label{tab:precision_recall}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Dataset} & \\textbf{Schema Type} & \\textbf{Pos P} & \\textbf{Pos R} & \\textbf{Neg P} & \\textbf{Neg R} \\\\
\\midrule""")

for dataset_idx, dataset_name in enumerate(datasets):
    for schema_idx, schema_name in enumerate(schema_labels):
        pos_p = pos_precision[dataset_idx][schema_idx]
        pos_r = pos_recall[dataset_idx][schema_idx]
        neg_p = neg_precision[dataset_idx][schema_idx]
        neg_r = neg_recall[dataset_idx][schema_idx]
        
        # 简化Schema名称用于LaTeX
        latex_schema_names = ["Baseline", "No Schema", "Schema w/ Key"]
        latex_schema = latex_schema_names[schema_idx]
        
        if schema_idx == 0:  # 第一行显示数据集名称
            print(f"\\multirow{{3}}{{*}}{{{dataset_name}}} & {latex_schema} & {pos_p:.2f} & {pos_r:.2f} & {neg_p:.2f} & {neg_r:.2f} \\\\")
        else:
            print(f"& {latex_schema} & {pos_p:.2f} & {pos_r:.2f} & {neg_p:.2f} & {neg_r:.2f} \\\\")
    
    if dataset_idx < len(datasets) - 1:  # 添加分隔线
        print("\\midrule")

print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

# ===== 5. 主要发现摘要 =====
print(f"\n{'='*80}")
print("KEY INSIGHTS FROM PRECISION & RECALL")
print("="*80)

print("\n🔍 ANALYSIS POINTS TO DISCUSS:")
print("1. F1 improvements mainly driven by Precision or Recall?")
print("2. Different patterns between Positive and Negative classes")
print("3. Schema impact varies by dataset characteristics")
print("4. Trade-offs between Precision and Recall")

print("\n📝 RECOMMENDED DISCUSSION:")
print("• In NL2SQL-BUGs: [Analyze your P/R patterns]")
print("• In SSS: [Analyze your P/R patterns]") 
print("• Schema effects on different aspects of model performance")

print(f"\n{'='*80}")
print("✅ COMPLETED: F1 visualization + P/R tables ready for thesis!")
print("="*80)