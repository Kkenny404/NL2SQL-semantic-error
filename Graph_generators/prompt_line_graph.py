import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# 实际数据
prompt_methods = ["Baseline", "CoT", "Self-reflection"]
datasets = ["NL2SQL-BUGs", "SSS"]

pos_f1 = [
    [0.71, 0.66, 0.71],  # NL2SQL-BUGs
    [0.53, 0.46, 0.43],  # SSS
]
neg_f1 = [
    [0.74, 0.75, 0.74],  # NL2SQL-BUGs
    [0.73, 0.73, 0.73],  # SSS
]

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# ===== 改进的水平条形图 =====
def create_improved_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # 计算相对于baseline的改善
    improvements = []
    labels = []
    colors = []
    
    color_map = {'NL2SQL-BUGs': '#2E86AB', 'SSS': '#F24236'}
    
    for i, dataset in enumerate(datasets):
        baseline_pos = pos_f1[i][0]
        baseline_neg = neg_f1[i][0]
        
        for j in range(1, len(prompt_methods)):
            pos_imp = ((pos_f1[i][j] - baseline_pos) / baseline_pos) * 100
            neg_imp = ((neg_f1[i][j] - baseline_neg) / baseline_neg) * 100
            
            improvements.extend([pos_imp, neg_imp])
            labels.extend([f"{dataset} - {prompt_methods[j]}\nPositive F1", 
                          f"{dataset} - {prompt_methods[j]}\nNegative F1"])
            colors.extend([color_map[dataset], color_map[dataset]])
    
    y_pos = np.arange(len(improvements))
    
    # 创建条形图，区分正负改进
    bars = []
    for i, imp in enumerate(improvements):
        if imp >= 0:
            bar = ax.barh(y_pos[i], imp, color=colors[i], alpha=0.7, 
                         edgecolor='black', linewidth=0.5)
        else:
            bar = ax.barh(y_pos[i], imp, color=colors[i], alpha=0.7, 
                         edgecolor='black', linewidth=0.5, hatch='///')
        bars.append(bar)
    
    # 添加数值标注
    for i, imp in enumerate(improvements):
        # 根据正负值调整标注位置
        offset = 0.5 if imp >= 0 else -0.5
        ha = 'left' if imp >= 0 else 'right'
        ax.text(imp + offset, y_pos[i], f'{imp:+.1f}%', 
               ha=ha, va='center', fontweight='bold', fontsize=10)
    
    # 设置轴标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Performance Change from Baseline (%)", fontsize=12, fontweight='bold')
    ax.set_title("Prompting Methods: Performance Change Analysis\n(Relative to Baseline)", 
                fontsize=16, fontweight='bold', pad=20)
    
    # 添加零线
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 添加改进/退化区域标记
    ax.text(5, len(improvements)-0.5, 'Improvement', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))
    ax.text(-5, len(improvements)-0.5, 'Degradation', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5))
    
    # 图例
    legend_elements = [
        patches.Patch(color=color_map['NL2SQL-BUGs'], label='NL2SQL-BUGs', alpha=0.7),
        patches.Patch(color=color_map['SSS'], label='SSS', alpha=0.7),
        patches.Patch(facecolor='white', edgecolor='black', hatch='///', 
                     label='Performance Degradation', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # 调整X轴范围以更好地显示数据
    ax.set_xlim(-25, 10)
    
    plt.tight_layout()
    plt.savefig("improved_prompt_methods_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# ===== 原始条形图（修正版本）=====
def create_original_barchart():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 设置颜色
    colors_pos = '#2E86AB'  # 深蓝色
    colors_neg = '#F24236'  # 红色
    
    bar_width = 0.35
    x = np.arange(len(prompt_methods))
    
    for i, ax in enumerate(axes):
        # 绘制柱状图
        bars_pos = ax.bar(x - bar_width/2, pos_f1[i], bar_width, 
                          label="Positive F1", color=colors_pos, 
                          alpha=0.8, edgecolor='black', linewidth=0.5)
        bars_neg = ax.bar(x + bar_width/2, neg_f1[i], bar_width, 
                          label="Negative F1", color=colors_neg, 
                          alpha=0.8, edgecolor='black', linewidth=0.5)
    
        # 添加数值标注
        for j in range(len(prompt_methods)):
            ax.text(x[j] - bar_width/2, pos_f1[i][j] + 0.01, f"{pos_f1[i][j]:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(x[j] + bar_width/2, neg_f1[i][j] + 0.01, f"{neg_f1[i][j]:.2f}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
        # 设置坐标轴
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_methods, fontsize=10, ha='center')
        ax.set_ylim(0.35, 0.80)  # 调整Y轴范围适应实际数据
        
        if i == 0:
            ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
        
        ax.set_title(f"{datasets[i]}", fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        ax.set_facecolor('#f8f9fa')
    
    # 图例和标题
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=2, frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    fig.suptitle("Impact of Prompting Methods on F1 Scores\n(Actual Performance Results)", 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.15, wspace=0.25)
    
    # 添加观察注释
    axes[0].annotate('Baseline performs best\nfor Positive F1', 
                    xy=(0, 0.71), xytext=(0.5, 0.77),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                    fontsize=10, ha='center', color='#333333')
    
    axes[1].annotate('Consistent degradation\nin SSS dataset', 
                    xy=(1, 0.46), xytext=(1.5, 0.55),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                    fontsize=10, ha='center', color='#333333')
    
    plt.savefig("prompt_methods_actual_results.png", dpi=300, bbox_inches='tight')
    plt.show()

# 执行两种可视化
print("=== 生成改进的对比图 ===")
create_improved_comparison()

print("\n=== 生成原始条形图（修正版）===")
create_original_barchart()

print("\n=== 数据分析 ===")
for i, dataset in enumerate(datasets):
    print(f"\n{dataset}:")
    baseline_pos = pos_f1[i][0]
    baseline_neg = neg_f1[i][0]
    
    for j, method in enumerate(prompt_methods):
        print(f"  {method}: Pos F1={pos_f1[i][j]:.2f}, Neg F1={neg_f1[i][j]:.2f}")
        if j > 0:  # 非baseline方法
            pos_change = ((pos_f1[i][j] - baseline_pos) / baseline_pos) * 100
            neg_change = ((neg_f1[i][j] - baseline_neg) / baseline_neg) * 100
            print(f"    Change: Pos {pos_change:+.1f}%, Neg {neg_change:+.1f}%")

print("\n=== 关键观察 ===")
print("1. NL2SQL-BUGs: Baseline在Positive F1上表现最佳")
print("2. SSS: 所有高级prompting方法都导致性能下降")
print("3. Negative F1在两个数据集上相对稳定")
print("4. 结果表明复杂prompting可能不适合这些特定任务")