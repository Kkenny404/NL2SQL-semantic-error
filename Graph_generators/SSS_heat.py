import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ==== 动态配置 ====
ALL_ERRORS_FILE = "Spider/error_explaination.json"
SSS_FILES = {
    "Schema with keys info (Baseline)": "results/SSS/eval_results/erros/Basline_error.json",
    "No Schema Provided": "results/SSS/eval_results/erros/No_schema_error.json",
    # "With Extra Parsed SQL": "results/SSS/eval_results/erros/parsed_sql_error.json",
    # "Simple (Baseline)": "results/SSS/eval_results/erros/Basline_error.json",
    # "Chain-of-Thought": "results/SSS/eval_results/erros/CoT_error.json",
    # "Self-Reflection": "results/SSS/eval_results/erros/selfF_error.json",
}

# 定义方法显示顺序
METHOD_ORDER = ["Simple (Baseline)", "Chain-of-Thought", "Self-Reflection"]

# SSS数据集专用配色方案
CATEGORY_COLORS = {
    'Function-related Errors': '#E53E3E',
    'Clause-related Errors': '#3182CE',
    'Attribute-related Errors': '#38A169',
    'Condition-related Errors': '#DD6B20',
    'Value-related Errors': '#805AD5',
    'Table-related Errors': '#D53F8C',
    'Operator-related Errors': '#319795',
    'Subquery-related Errors': '#A0AEC0',
    'Other Errors': '#4A5568'
}

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

# ==== 动态数据读取和预处理 ====
def load_and_process_data_dynamically():
    """动态加载和处理数据，自动提取所有可用的方法和错误类型"""
    
    # 首先从一个示例文件中获取错误类型结构
    sample_file = list(SSS_FILES.values())[0]
    with open(sample_file) as f:
        sample_data = json.load(f)
    
    # 自动提取大类和子错误类型
    sub_error_to_category = {}
    ordered_categories = []
    ordered_sub_errors = []
    
    for big_category, sub_errors_dict in sample_data.items():
        if big_category == "average_recall":  # 跳过总体统计
            continue
        ordered_categories.append(big_category)
        for sub_error in sub_errors_dict.keys():
            if sub_error != "average_recall":
                sub_error_to_category[sub_error] = big_category
                ordered_sub_errors.append(sub_error)
    
    # 使用预定义的方法顺序，只包含实际存在的方法
    available_methods = [method for method in METHOD_ORDER if method in SSS_FILES.keys()]
    
    # 读取所有方法的数据
    all_methods_data = {}
    data = []
    
    for method, path in SSS_FILES.items():
        with open(path) as f:
            method_data = json.load(f)
        all_methods_data[method] = method_data
        
        for big_category, sub_errors_dict in method_data.items():
            if big_category == "average_recall":
                continue
            for sub_error, metrics in sub_errors_dict.items():
                if sub_error != "average_recall":
                    recall = metrics.get("recall", None)
                    data.append({
                        "Method": method,
                        "Error Type": sub_error,
                        "Big Category": big_category,
                        "Recall": recall
                    })
    
    df = pd.DataFrame(data)
    
    # 返回处理后的数据和元信息
    return df, sub_error_to_category, ordered_categories, ordered_sub_errors, available_methods, all_methods_data

# ==== 动态生成颜色方案 ====
def generate_dynamic_colors(categories, base_colors=None):
    """为新的类别动态生成颜色"""
    if base_colors is None:
        base_colors = CATEGORY_COLORS
    
    # 如果有新的类别，为它们生成颜色
    import matplotlib.colors as mcolors
    available_colors = list(mcolors.TABLEAU_COLORS.values())
    
    color_scheme = base_colors.copy()
    used_colors = set(base_colors.values())
    
    for category in categories:
        if category not in color_scheme:
            # 为新类别分配颜色
            for color in available_colors:
                if color not in used_colors:
                    color_scheme[category] = color
                    used_colors.add(color)
                    break
    
    return color_scheme

# 加载数据
df, sub_error_to_category, ordered_categories, ordered_sub_errors, available_methods, all_methods_data = load_and_process_data_dynamically()

# 动态生成颜色方案
DYNAMIC_CATEGORY_COLORS = generate_dynamic_colors(ordered_categories)

print(f"Detected Methods (in order): {available_methods}")
print(f"Detected Categories: {ordered_categories}")
print(f"Total Sub-errors: {len(ordered_sub_errors)}")

# ==== 图1: 动态大类概览热力图 ====
def create_dynamic_category_overview():
    category_data = []
    
    # 动态处理所有方法
    for method in available_methods:
        method_data = df[df['Method'] == method]
        for category in ordered_categories:
            cat_data = method_data[method_data['Big Category'] == category]
            avg_recall = cat_data['Recall'].mean()
            category_data.append({
                'Method': method,
                'Category': category,
                'Average Recall': avg_recall
            })
    
    cat_df = pd.DataFrame(category_data)
    cat_pivot = cat_df.pivot(index="Category", columns="Method", values="Average Recall")
    
    # 重新排序列为指定顺序
    cat_pivot = cat_pivot.reindex(columns=available_methods)
    
    # 按平均性能排序
    cat_pivot['mean_perf'] = cat_pivot.mean(axis=1)
    cat_pivot_sorted = cat_pivot.sort_values('mean_perf', ascending=True)
    cat_pivot_for_plot = cat_pivot_sorted.drop('mean_perf', axis=1)
    
    # 动态调整图形大小
    fig_width = max(10, len(available_methods) * 2)
    fig_height = max(8, len(ordered_categories) * 0.8)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 使用动态范围
    vmin = cat_pivot_for_plot.min().min() * 0.9
    vmax = cat_pivot_for_plot.max().max() * 1.05
    
    im = sns.heatmap(cat_pivot_for_plot, 
                     annot=True, 
                     cmap="RdYlGn", 
                     cbar=True,
                     linewidths=2, 
                     fmt=".3f", 
                     vmin=vmin, 
                     vmax=vmax,
                     annot_kws={"size": 13, "weight": "bold"},
                     cbar_kws={"shrink": 0.8, "label": "Average Recall"},
                     ax=ax)
    
    ax.set_title("Baseline -- Error Category Performance Overview\n (SSS Dataset)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Error Category", fontsize=13, fontweight='bold')
    ax.set_xlabel("Method", fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    
    plt.tight_layout()
    return fig

# ==== 图2: 动态详细热力图 ====
def create_dynamic_detailed_heatmap():
    pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
    
    # 重新排序列为指定顺序：Baseline, CoT, Self-Reflection
    pivot_sub = pivot_sub.reindex(columns=available_methods)
    pivot_sub = pivot_sub.reindex(ordered_sub_errors)
    
    # 按大类分组并排序
    grouped_errors = []
    category_positions = {}
    current_pos = 0
    
    for category in ordered_categories:
        category_errors = [err for err in ordered_sub_errors if sub_error_to_category.get(err) == category]
        if category_errors:
            category_subset = pivot_sub.loc[category_errors]
            category_subset['mean_perf'] = category_subset.mean(axis=1)
            category_subset_sorted = category_subset.sort_values('mean_perf', ascending=True)
            grouped_errors.extend(category_subset_sorted.index.tolist())
            
            category_positions[category] = {
                'start': current_pos,
                'end': current_pos + len(category_errors),
                'mid': current_pos + len(category_errors) / 2
            }
            current_pos += len(category_errors)
    
    pivot_sub_grouped = pivot_sub.reindex(grouped_errors)
    
    # 动态调整图形大小
    fig_width = max(8, len(available_methods) * 3)
    fig_height = max(12, len(grouped_errors) * 0.4)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 动态确定颜色范围
    vmin = pivot_sub_grouped.min().min() * 0.9
    vmax = pivot_sub_grouped.max().max() * 1.05
    
    im = sns.heatmap(pivot_sub_grouped, 
                     annot=True, 
                     cmap="RdYlGn", 
                     cbar=True,
                     linewidths=1, 
                     fmt=".2f", 
                     vmin=vmin, 
                     vmax=vmax,
                     annot_kws={"size": 10},
                     cbar_kws={"shrink": 0.8, "label": "Recall Score"},
                     ax=ax)
    
    # 添加大类分割线
    for category, pos_info in category_positions.items():
        if pos_info['start'] > 0:
            ax.axhline(y=pos_info['start'], color='white', linewidth=4)
            ax.axhline(y=pos_info['start'], color='black', linewidth=2)
    
    # 设置y轴标签颜色
    y_colors = []
    for error in grouped_errors:
        category = sub_error_to_category.get(error, "Unknown")
        y_colors.append(DYNAMIC_CATEGORY_COLORS.get(category, 'black'))
    
    for i, (label, color) in enumerate(zip(ax.get_yticklabels(), y_colors)):
        label.set_color(color)
        label.set_fontweight('bold')
        label.set_fontsize(10)
    
    ax.set_title("Prompting Methods -- Detailed Error Type Recall Analysis\n (SSS Dataset)", 
                fontsize=16, fontweight='bold', pad=25)
    ax.set_ylabel("Sub Error Type", fontsize=13, fontweight='bold')
    ax.set_xlabel("Method", fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)

    # 添加类别标签
    ax.text(-0.6, 0.5, "Error Categories",
            ha='center', va='center', rotation='vertical',
            fontsize=13, fontweight='bold',
            transform=ax.transAxes)
            
    for category, pos_info in category_positions.items():
        y_center = pos_info['start'] + (pos_info['end'] - pos_info['start']) / 2 + 0.5
        category_short = category.replace('-related', '').replace(' Errors', '')
        
        ax.text(-0.5, y_center, category_short,
                rotation=0, ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=DYNAMIC_CATEGORY_COLORS.get(category, 'black'),
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor='white', 
                         edgecolor=DYNAMIC_CATEGORY_COLORS.get(category, 'black'),
                         linewidth=1.5, alpha=0.9))

    plt.subplots_adjust(left=0.25)
    return fig

# ==== 图3: 动态性能对比分析 ====
def create_dynamic_comparison_analysis():
    """创建动态的性能对比分析，支持任意数量的方法"""
    
    pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
    # 重新排序列
    pivot_sub = pivot_sub.reindex(columns=available_methods)
    
    # 如果只有一个方法，显示绝对性能分析
    if len(available_methods) == 1:
        return create_single_method_analysis(pivot_sub)
    
    # 多方法对比分析
    return create_multi_method_comparison(pivot_sub)

def create_single_method_analysis(pivot_sub):
    """单方法的性能分析"""
    method = available_methods[0]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 性能分布直方图
    ax1.hist(pivot_sub[method], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Recall Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{method}: Performance Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 按类别的性能
    category_data = []
    for error_type in pivot_sub.index:
        category_data.append({
            'Error Type': error_type,
            'Performance': pivot_sub.loc[error_type, method],
            'Category': sub_error_to_category.get(error_type, 'Other')
        })
    
    cat_df = pd.DataFrame(category_data)
    cat_avg = cat_df.groupby('Category')['Performance'].mean().sort_values()
    
    colors = [DYNAMIC_CATEGORY_COLORS.get(cat, 'gray') for cat in cat_avg.index]
    bars = ax2.barh(range(len(cat_avg)), cat_avg.values, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(cat_avg)))
    ax2.set_yticklabels([cat.replace(' Errors', '') for cat in cat_avg.index])
    ax2.set_xlabel('Average Recall')
    ax2.set_title(f'{method}: Performance by Category', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, cat_avg.values)):
        ax2.text(val + 0.01, i, f'{val:.3f}', va='center', ha='left', fontweight='bold')
    
    # Top/Bottom 错误类型
    top_errors = pivot_sub.nlargest(10, method)
    bottom_errors = pivot_sub.nsmallest(10, method)
    
    ax3.barh(range(len(top_errors)), top_errors[method], color='green', alpha=0.7)
    ax3.set_yticks(range(len(top_errors)))
    ax3.set_yticklabels([err[:30] + '...' if len(err) > 30 else err for err in top_errors.index], fontsize=8)
    ax3.set_title(f'{method}: Top 10 Performance', fontweight='bold')
    ax3.set_xlabel('Recall Score')
    
    ax4.barh(range(len(bottom_errors)), bottom_errors[method], color='red', alpha=0.7)
    ax4.set_yticks(range(len(bottom_errors)))
    ax4.set_yticklabels([err[:30] + '...' if len(err) > 30 else err for err in bottom_errors.index], fontsize=8)
    ax4.set_title(f'{method}: Bottom 10 Performance', fontweight='bold')
    ax4.set_xlabel('Recall Score')
    
    plt.tight_layout()
    return fig

def create_multi_method_comparison(pivot_sub):
    """多方法对比分析"""
    
    # 使用第一个和最后一个方法进行对比（通常是Baseline vs 最佳方法）
    method1 = available_methods[0]  # Baseline
    method2 = available_methods[-1]  # 最后一个方法（通常是最好的）
    
    differences = {}
    for error_type in pivot_sub.index:
        val1 = pivot_sub.loc[error_type, method1]
        val2 = pivot_sub.loc[error_type, method2]
        
        differences[error_type] = {
            f'{method2} vs {method1}': val2 - val1,
            'Category': sub_error_to_category.get(error_type, 'Other'),
            method1: val1,
            method2: val2
        }
    
    # 转换为DataFrame
    diff_data = []
    for error_type, diffs in differences.items():
        diff_data.append({
            'Error Type': error_type,
            'Difference': diffs[f'{method2} vs {method1}'],
            'Category': diffs['Category'],
            method1: diffs[method1],
            method2: diffs[method2]
        })
    
    diff_df = pd.DataFrame(diff_data)
    
    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 性能差异分布
    for category in ordered_categories:
        cat_data = diff_df[diff_df['Category'] == category]
        if not cat_data.empty:
            ax1.scatter(cat_data[method1], cat_data['Difference'],
                       color=DYNAMIC_CATEGORY_COLORS.get(category, 'gray'),
                       alpha=0.7, s=80, label=category.replace(' Errors', ''))
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel(f'{method1} Performance')
    ax1.set_ylabel(f'Performance Change ({method2} vs {method1})')
    ax1.set_title(f'Performance Change vs {method1} Performance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # 子图2: 按类别聚合的改进/退化
    cat_differences = diff_df.groupby('Category')['Difference'].mean().sort_values()
    colors = [DYNAMIC_CATEGORY_COLORS.get(cat, 'gray') for cat in cat_differences.index]
    
    bars = ax2.barh(range(len(cat_differences)), cat_differences.values, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_yticks(range(len(cat_differences)))
    ax2.set_yticklabels([cat.replace(' Errors', '') for cat in cat_differences.index])
    ax2.set_xlabel('Average Performance Change')
    ax2.set_title(f'Average Change by Category\n({method2} vs {method1})', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, cat_differences.values)):
        ax2.text(val + (0.005 if val >= 0 else -0.005), i, f'{val:.3f}',
                va='center', ha='left' if val >= 0 else 'right', fontweight='bold')
    
    # 子图3: 改进vs退化统计
    improved = (diff_df['Difference'] > 0).sum()
    degraded = (diff_df['Difference'] < 0).sum()
    unchanged = (diff_df['Difference'] == 0).sum()
    
    counts = [improved, degraded, unchanged]
    labels = ['Improved', 'Degraded', 'Unchanged']
    colors_pie = ['green', 'red', 'gray']
    
    ax3.pie(counts, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
           startangle=90, textprops={'fontweight': 'bold'})
    ax3.set_title('Performance Change Distribution', fontweight='bold')
    
    # 子图4: 统计摘要
    top_improvements = diff_df.nlargest(5, 'Difference')
    top_degradations = diff_df.nsmallest(5, 'Difference')
    
    ax4.axis('off')
    summary_text = f"""PERFORMANCE COMPARISON SUMMARY

Overall Performance:
• {method1} Average: {diff_df[method1].mean():.3f}
• {method2} Average: {diff_df[method2].mean():.3f}
• Overall Change: {diff_df['Difference'].mean():.3f}

Top 5 Improvements ({method2} vs {method1}):
"""
    
    for _, row in top_improvements.iterrows():
        summary_text += f"  • {row['Error Type'][:40]}...: +{row['Difference']:.3f}\n"
    
    summary_text += f"\nTop 5 Degradations:\n"
    for _, row in top_degradations.iterrows():
        summary_text += f"  • {row['Error Type'][:40]}...: {row['Difference']:.3f}\n"
    
    summary_text += f"""
Statistics:
• Improved: {improved} error types
• Degraded: {degraded} error types  
• Unchanged: {unchanged} error types
• Largest improvement: +{diff_df['Difference'].max():.3f}
• Largest degradation: {diff_df['Difference'].min():.3f}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

# ==== 生成所有动态图表 ====
if __name__ == "__main__":
    print("Generating dynamic visualizations...")
    print(f"Processing {len(available_methods)} methods in order: {available_methods}")
    print(f"Processing {len(ordered_categories)} categories: {ordered_categories}")
    print(f"Processing {len(ordered_sub_errors)} sub-error types")
    
    # 创建图表
    fig1 = create_dynamic_category_overview()
    fig1.savefig('dynamic_category_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig2 = create_dynamic_detailed_heatmap()
    fig2.savefig('dynamic_detailed_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig3 = create_dynamic_comparison_analysis()
    fig3.savefig('dynamic_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All dynamic visualizations have been generated and saved!")
    
    # 打印统计信息
    print("\n=== Dataset Statistics ===")
    pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
    pivot_sub = pivot_sub.reindex(columns=available_methods)
    
    for method in available_methods:
        avg_performance = pivot_sub[method].mean()
        print(f"{method} Average: {avg_performance:.3f}")
    
    if len(available_methods) > 1:
        # 计算相对于第一个方法（Baseline）的改进
        baseline_method = available_methods[0]
        for method in available_methods[1:]:
            avg_diff = (pivot_sub[method] - pivot_sub[baseline_method]).mean()
            print(f"{method} vs {baseline_method} Average Change: {avg_diff:+.3f}")