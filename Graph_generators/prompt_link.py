import matplotlib.pyplot as plt
import numpy as np

# 实际数据来自表格
prompt_methods = ["Simple (baseline)", "CoT", "Self-reflection"]
datasets = ["NL2SQL-BUGs", "SSS"]

# 从你的表格提取的关键检测指标
data = {
    "NL2SQL-BUGs": {
        "Pos_P": [0.77, 0.81, 0.77],  # 正确检测精度
        "Neg_P": [0.70, 0.66, 0.70],  # 错误检测精度  
        "Neg_R": [0.80, 0.86, 0.79],  # 错误检测召回率（最重要！）
    },
    "SSS": {
        "Pos_P": [0.68, 0.70, 0.72],  # 正确检测精度
        "Neg_P": [0.65, 0.63, 0.62],  # 错误检测精度
        "Neg_R": [0.84, 0.88, 0.90],  # 错误检测召回率（最重要！）
    }
}

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})

# ===== 检测性能专用图表 =====
def create_detection_focused_chart():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 保留原始指标名称
    metrics = {
        "Negative Recall": "Neg_R",      # 最重要：能发现多少错误
        "Negative Precision": "Neg_P",   # 报错时的准确性
        "Positive Precision": "Pos_P"    # 说正确时的可靠性
    }
    
    colors = ['#D32F2F', '#FF6F00', '#388E3C']  # 红色(错误检测率), 橙色(错误精度), 绿色(正确精度)
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    
    x = np.arange(len(prompt_methods))
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # 为每个指标绘制线条
        for idx, (metric_name, metric_key) in enumerate(metrics.items()):
            values = data[dataset][metric_key]
            
            # Negative Recall用更粗的线条突出
            linewidth = 4 if metric_key == "Neg_R" else 2.5
            markersize = 10 if metric_key == "Neg_R" else 7
            
            ax.plot(x, values, marker=markers[idx], linewidth=linewidth, 
                   markersize=markersize, color=colors[idx], label=metric_name, 
                   alpha=0.9, linestyle=linestyles[idx])
            
            # 添加数值标注
            for j, val in enumerate(values):
                # Negative Recall的标注更突出
                fontweight = 'bold' if metric_key == "Neg_R" else 'normal'
                fontsize = 12 if metric_key == "Neg_R" else 12
                
                # 调整标注位置避免重叠
                offset_y = 18 if idx == 0 else (10 if idx == 1 else -16)
                
                ax.annotate(f'{val:.2f}', (j, val), 
                          textcoords="offset points", xytext=(0, offset_y), 
                          ha='center', fontweight=fontweight, fontsize=fontsize,
                          color=colors[idx])
        
        # 设置子图属性
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_methods, fontsize=12)
        ax.set_ylabel("Performance Score", fontsize=12, fontweight='bold')
        ax.set_title(f"{dataset}", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0.60, 0.92)  # 更紧凑的Y轴范围，突出差异
        
        # 高亮最佳Negative Recall
        neg_r_values = data[dataset]["Neg_R"]
        best_idx = np.argmax(neg_r_values)
        best_val = neg_r_values[best_idx]
        ax.scatter(best_idx, best_val, s=120, color='gold', 
                  edgecolor='darkred', linewidth=2.5, zorder=10, alpha=0.9)
        
        # 添加最佳性能标注
        ax.annotate(f'Best: {best_val:.2f}', 
                   xy=(best_idx, best_val), xytext=(best_idx+0.25, best_val+0.03),
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                   fontsize=12, ha='center', color='darkred', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.8))
    
    # 图例 - 突出Negative Recall
    handles, labels = axes[0].get_legend_handles_labels()
    # 重新排序，把Negative Recall放在第一位
    neg_recall_idx = labels.index("Negative Recall")
    new_handles = [handles[neg_recall_idx]] + [h for i, h in enumerate(handles) if i != neg_recall_idx]
    new_labels = [labels[neg_recall_idx]] + [l for i, l in enumerate(labels) if i != neg_recall_idx]
    
    fig.legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
              ncol=3, frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    plt.suptitle("Prompting Methods Performance (Simple, CoT, Self-refelction): Precision and Recall Analysis\n", 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    plt.savefig("prompting_methods_precision_recall.png", dpi=300, bbox_inches='tight')
    plt.show()

# ===== 错误检测率对比图 =====
def create_error_detection_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 专注于错误检测率对比
    colors = ['#2E86AB', '#F24236']  # NL2SQL-BUGs: 蓝色, SSS: 红色
    bar_width = 0.25
    x = np.arange(len(prompt_methods))
    
    # 绘制错误检测率对比
    for i, dataset in enumerate(datasets):
        neg_r_values = data[dataset]["Neg_R"]
        bars = ax.bar(x + i*bar_width, neg_r_values, bar_width, 
                     label=f"{dataset}", color=colors[i], alpha=0.8,
                     edgecolor='black', linewidth=0.5)
        
        # 添加数值标注
        for j, val in enumerate(neg_r_values):
            ax.text(x[j] + i*bar_width, val + 0.01, f"{val:.2f}", 
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 设置图表属性
    ax.set_xticks(x + bar_width/2)
    ax.set_xticklabels(prompt_methods, fontsize=13)
    ax.set_ylabel("Error Detection Rate (Negative Recall)", fontsize=12, fontweight='bold')
    ax.set_title("SQL Error Detection Rate by Prompting Method\n(Most Critical Metric for Detection Tasks)", 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.75, 0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 图例
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    # 添加关键观察
    ax.text(0.5, 0.92, "Higher is Better\n(Can detect more errors)", 
           ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig("error_detection_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# 执行可视化
print("=== 生成检测任务专用性能图 ===")
create_detection_focused_chart()

print("\n=== 生成错误检测率对比图 ===")
create_error_detection_comparison()

print("\n=== 检测任务关键发现 ===")
for dataset in datasets:
    print(f"\n{dataset} - 错误检测能力排名:")
    neg_r_values = data[dataset]["Neg_R"]
    for i, (method, rate) in enumerate(zip(prompt_methods, neg_r_values)):
        print(f"  {i+1}. {method}: {rate:.2f} (能发现{rate*100:.0f}%的错误)")

print("\n=== 实际应用建议 ===")
print("• CoT在两个数据集上都有最高的错误检测率")
print("• Self-reflection在SSS上达到90%的错误检测率")
print("• 对于SQL错误检测任务，推荐优先考虑Negative Recall最高的方法")