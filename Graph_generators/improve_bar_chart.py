import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ==== 配置 ====
ALL_ERRORS_FILE = "Spider/error_explaination.json"
FILES = {
    # "Schema without Keys Info (Baseline)": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
    # "Schema With Keys Info": "results/eval_result/errors/KeyInfo_GPT_results_error.json",
    # "No Schema Provided": "results/eval_result/errors/no_schema_error.json",
    # 可以添加更多方法
    # "Original SQL Only (Baseline)": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
    # "With Extra Parsed SQL": "results/eval_result/errors/parserBUGS_GPT4.1_error_20250804_140520.json",
    "Simple (Baseline)": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
    "Chain-of-Thought": "results/eval_result/errors/CoT_error.json",
    "Self-Reflection": "results/eval_result/errors/selfF_error.json",

}

# ==== 名称映射（可选，用于美化显示） ====
name_mapping = {
    "ASC_DESC": "ASC/DESC",
    "DateTime Functions": "Date/Time Functions",
    "Comparison Operator": "Comparison Operator Mismatch",
    "Logical Operator": "Logical Operator Mismatch",
    "Others": "Other"
}

def generate_category_colors(categories):
    """动态生成类别颜色映射"""
    colors = [
        '#D32F2F',  # 深红色
        '#1976D2',  # 深蓝色
        '#388E3C',  # 深绿色
        '#F57C00',  # 深橙色
        '#7B1FA2',  # 深紫色
        '#C2185B',  # 深粉色
        '#00796B',  # 深青色
        '#5D4037',  # 深棕色
        '#455A64',  # 深灰色
        '#E65100',  # 深橙红
        '#4527A0',  # 深靛色
        '#2E7D32',  # 深森林绿
    ]
    
    return {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

def generate_comparison_colors(methods, baseline_method):
    """动态生成对比方法颜色"""
    colors = [
        "#AC6ECB",  # 深兰花紫
        '#36C5F0',  # 天蓝色
        '#4682B4',  # 钢蓝色
        "#AC6ECB",  # 深兰花紫
        '#4169E1',  # 皇家蓝
        '#2E8B57',  # 海绿色
        '#FF8C00',  # 深橙色
        '#DC143C',  # 深红色
        '#228B22',  # 森林绿
        '#FF6347',  # 番茄红

    ]
    
    comparison_colors = {}
    color_idx = 0
    for method in methods:
        if method != baseline_method:
            comparison_colors[method] = {
                'color': colors[color_idx % len(colors)], 
                'label': method
            }
            color_idx += 1
    
    return comparison_colors

def identify_baseline_method(methods):
    """智能识别baseline方法"""
    # 检查是否有包含baseline关键词的方法
    for method in methods:
        if any(keyword in method.lower() for keyword in ['baseline', 'base', 'reference']):
            return method
    
    # 如果没有明显的baseline，返回第一个方法
    return methods[0]

def find_metric_key(sample_metrics):
    """动态找到性能指标键"""
    possible_keys = ["recall", "accuracy", "f1", "f1_score", "score", "performance", "value", "precision"]
    skip_keys = {"average_recall", "avg_recall", "total", "summary", "count"}
    
    if isinstance(sample_metrics, dict):
        for key in possible_keys:
            if key in sample_metrics and key not in skip_keys:
                return key
        
        # 如果都没找到，返回第一个数值型键
        for key, value in sample_metrics.items():
            if isinstance(value, (int, float)) and key not in skip_keys:
                return key
    
    return "recall"  # 默认值

def load_and_process_data():
    """加载和处理错误数据 - 动态适配"""
    # 读取全集错误类型（如果存在）
    try:
        with open(ALL_ERRORS_FILE) as f:
            categories = json.load(f)
    except FileNotFoundError:
        categories = None
    
    # 从第一个文件获取数据结构
    sample_file = list(FILES.values())[0]
    with open(sample_file) as f:
        sample_data = json.load(f)
    
    # 动态识别跳过的键
    skip_keys = {"average_recall", "avg_recall", "total", "summary", "count", "overall"}
    
    # 建立子错误到大类的映射
    sub_error_to_category = {}
    ordered_categories = []
    ordered_sub_errors = []
    metric_key = None
    
    for big_category, content in sample_data.items():
        if big_category in skip_keys:
            continue
            
        if isinstance(content, dict):
            ordered_categories.append(big_category)
            for sub_error, metrics in content.items():
                if sub_error not in skip_keys:
                    sub_error_to_category[sub_error] = big_category
                    ordered_sub_errors.append(sub_error)
                    
                    # 动态识别性能指标键
                    if metric_key is None and isinstance(metrics, dict):
                        metric_key = find_metric_key(metrics)
        elif isinstance(content, (int, float)):
            # 平面结构
            ordered_categories.append(big_category)
            ordered_sub_errors.append(big_category)
            sub_error_to_category[big_category] = big_category
    
    if metric_key is None:
        metric_key = "recall"  # 默认值
    
    # 读取所有方法的结果并整合
    data = []
    methods = list(FILES.keys())
    baseline_method = identify_baseline_method(methods)
    
    for method, path in FILES.items():
        with open(path) as f:
            res = json.load(f)
        
        for big_category, content in res.items():
            if big_category in skip_keys:
                continue
                
            if isinstance(content, dict):
                for sub_error, metrics in content.items():
                    if sub_error not in skip_keys:
                        if isinstance(metrics, dict):
                            metric_value = metrics.get(metric_key, None)
                        else:
                            metric_value = metrics
                        
                        data.append({
                            "Method": method,
                            "Error Type": sub_error,
                            "Big Category": big_category,
                            "Metric": metric_value
                        })
            else:
                # 平面结构
                data.append({
                    "Method": method,
                    "Error Type": big_category,
                    "Big Category": big_category,
                    "Metric": content
                })
    
    # 生成颜色映射
    category_colors = generate_category_colors(ordered_categories)
    comparison_colors = generate_comparison_colors(methods, baseline_method)
    
    return (pd.DataFrame(data), sub_error_to_category, ordered_categories, 
            ordered_sub_errors, baseline_method, category_colors, comparison_colors, metric_key)

def create_baseline_comparison_chart(df, ordered_categories, baseline_method, 
                                   category_colors, comparison_colors, metric_key):
    """创建以Baseline为基准的对比图 - 动态适配"""
    methods = df['Method'].unique().tolist()
    comparison_methods = [m for m in methods if m != baseline_method]
    
    # 计算大类平均性能
    category_data = []
    for method in methods:
        method_data = df[df['Method'] == method]
        for category in ordered_categories:
            cat_data = method_data[method_data['Big Category'] == category]
            avg_metric = cat_data['Metric'].mean()
            category_data.append({
                'Method': method,
                'Category': category,
                'Average Metric': avg_metric
            })
    
    cat_df = pd.DataFrame(category_data)
    
    # 获取baseline性能
    baseline_df = cat_df[cat_df['Method'] == baseline_method].set_index('Category')['Average Metric']
    
    # 计算相对于baseline的差异
    comparison_data = []
    for method in comparison_methods:
        method_df = cat_df[cat_df['Method'] == method].set_index('Category')['Average Metric']
        for category in ordered_categories:
            if category in baseline_df.index and category in method_df.index:
                baseline_val = baseline_df[category]
                method_val = method_df[category]
                if baseline_val > 0:  # 避免除零
                    difference = method_val - baseline_val
                    percentage_diff = (difference / baseline_val) * 100
                    comparison_data.append({
                        'Method': method,
                        'Category': category,
                        'Difference': difference,
                        'Percentage_Diff': percentage_diff,
                        'Baseline_Value': baseline_val,
                        'Method_Value': method_val
                    })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # 按baseline性能排序大类（从低到高）
    baseline_sorted = baseline_df.sort_values(ascending=True)
    ordered_cats = baseline_sorted.index.tolist()
    
    # 动态简化大类名称
    category_display_names = {}
    for cat in ordered_cats:
        display_name = cat
        suffixes_to_remove = ['-Related Errors', ' Errors', '-related', '_error', '_errors', '-Related']
        for suffix in suffixes_to_remove:
            if display_name.endswith(suffix):
                display_name = display_name[:-len(suffix)]
                break
        category_display_names[cat] = display_name
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 设置y轴位置
    y_pos = np.arange(len(ordered_cats))
    height = 0.8 / len(comparison_methods) if len(comparison_methods) > 1 else 0.6
    
    # 绘制对比条形图
    for i, method in enumerate(comparison_methods):
        method_data = comp_df[comp_df['Method'] == method]
        
        differences = []
        baseline_values = []
        method_values = []
        
        for cat in ordered_cats:
            cat_data = method_data[method_data['Category'] == cat]
            if not cat_data.empty:
                diff = cat_data['Difference'].iloc[0]
                differences.append(diff)
                baseline_values.append(cat_data['Baseline_Value'].iloc[0])
                method_values.append(cat_data['Method_Value'].iloc[0])
            else:
                differences.append(0)
                baseline_values.append(0)
                method_values.append(0)
        
        # 计算y位置偏移
        if len(comparison_methods) > 1:
            y_offset = (i - (len(comparison_methods) - 1) / 2) * height
        else:
            y_offset = 0
        
        # 绘制水平条形图
        bars = ax.barh(y_pos + y_offset, differences, height,
                      label=comparison_colors[method]['label'],
                      color=comparison_colors[method]['color'],
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=0.8)
        
        # 添加数值标签
        for j, (bar, diff, baseline_val, method_val) in enumerate(zip(bars, differences, baseline_values, method_values)):
            if abs(diff) > 0.001:  # 只显示有意义的差异
                # 计算标签位置
                label_x = bar.get_width() + (0.01 if diff >= 0 else -0.01)
                ha = 'left' if diff >= 0 else 'right'
                
                # 创建标签文本
                pct_change = (diff / baseline_val * 100) if baseline_val > 0 else 0
                label_text = f'{diff:+.3f}\n({pct_change:+.1f}%)'
                
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       label_text, ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # 添加baseline参考线（x=0）
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Baseline')
    
    # 设置标签和标题
    metric_display = metric_key.title().replace('_', ' ')
    ax.set_xlabel(f'Difference from Baseline (Nagetive {metric_display})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error Categories', fontsize=14, fontweight='bold')
    
    # 动态生成标题
    baseline_display = baseline_method.replace('\n', ' ')
    ax.set_title(f'Prompting Methods - Performance Comparison Relative to Baseline\n(NL2SQL-BUGs Dataset)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 设置y轴标签并着色
    ax.set_yticks(y_pos)
    y_labels = [category_display_names[cat] for cat in ordered_cats]
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    
    # 为y轴标签着色
    for i, (label, cat) in enumerate(zip(ax.get_yticklabels(), ordered_cats)):
        label.set_color(category_colors.get(cat, 'black'))
    
    # 设置x轴
    if len(comp_df) > 0:
        max_abs_diff = max([abs(d) for d in comp_df['Difference']] + [0.1])
        ax.set_xlim(-max_abs_diff * 1.3, max_abs_diff * 1.3)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加图例
    legend_elements = []
    for method in comparison_methods:
        legend_elements.append(
            plt.Rectangle((0,0),1,1, facecolor=comparison_colors[method]['color'], 
                         alpha=0.8, label=comparison_colors[method]['label'])
        )
    legend_elements.append(
        plt.Line2D([0], [0], color='black', linewidth=2, label=f'{baseline_method}')
    )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)
    
    # 添加baseline值信息
    baseline_text = f"Baseline Performance ({baseline_method.split('(')[0].strip()}):\n"
    for cat in ordered_cats:
        baseline_val = baseline_sorted[cat]
        cat_display = category_display_names[cat]
        baseline_text += f"{cat_display}: {baseline_val:.3f}\n"
    
    ax.text(0.02, 0.98, baseline_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return comp_df, ordered_cats, baseline_sorted

def create_detailed_baseline_comparison(df, sub_error_to_category, ordered_cats, baseline_sorted,
                                      baseline_method, comparison_colors):
    """创建详细的子错误类型baseline对比图 - 动态适配"""
    methods = df['Method'].unique().tolist()
    comparison_methods = [m for m in methods if m != baseline_method]
    
    # 选择表现最差的N个大类进行详细分析
    n_worst = min(3, len(baseline_sorted))
    worst_categories = list(baseline_sorted.head(n_worst).index)
    
    fig, axes = plt.subplots(len(worst_categories), 1, figsize=(16, 8*len(worst_categories)))
    if len(worst_categories) == 1:
        axes = [axes]
    
    for idx, category in enumerate(worst_categories):
        ax = axes[idx]
        
        # 获取该大类下的所有子错误
        sub_errors = [err for err, cat in sub_error_to_category.items() if cat == category]
        category_data = df[df['Big Category'] == category]
        
        # 获取baseline性能
        baseline_data = category_data[category_data['Method'] == baseline_method]
        baseline_sub_performance = baseline_data.set_index('Error Type')['Metric']
        
        # 计算对比数据
        comparison_sub_data = []
        for method in comparison_methods:
            method_data = category_data[category_data['Method'] == method]
            method_sub_performance = method_data.set_index('Error Type')['Metric']
            
            for sub_error in sub_errors:
                if sub_error in baseline_sub_performance.index and sub_error in method_sub_performance.index:
                    baseline_val = baseline_sub_performance[sub_error]
                    method_val = method_sub_performance[sub_error]
                    difference = method_val - baseline_val
                    
                    comparison_sub_data.append({
                        'Method': method,
                        'Sub_Error': sub_error,
                        'Difference': difference,
                        'Baseline_Value': baseline_val,
                        'Method_Value': method_val
                    })
        
        comp_sub_df = pd.DataFrame(comparison_sub_data)
        
        # 按baseline性能排序子错误
        sorted_sub_errors = baseline_sub_performance.sort_values(ascending=True).index.tolist()
        sorted_sub_errors = [err for err in sorted_sub_errors if err in sub_errors]
        
        if not sorted_sub_errors:
            continue
        
        # 设置y轴位置
        y_pos = np.arange(len(sorted_sub_errors))
        height = 0.8 / len(comparison_methods) if len(comparison_methods) > 1 else 0.6
        
        # 绘制对比条形图
        for i, method in enumerate(comparison_methods):
            method_data = comp_sub_df[comp_sub_df['Method'] == method]
            
            differences = []
            for sub_error in sorted_sub_errors:
                sub_data = method_data[method_data['Sub_Error'] == sub_error]
                if not sub_data.empty:
                    diff = sub_data['Difference'].iloc[0]
                    differences.append(diff)
                else:
                    differences.append(0)
            
            # 计算y位置偏移
            if len(comparison_methods) > 1:
                y_offset = (i - (len(comparison_methods) - 1) / 2) * height
            else:
                y_offset = 0
            
            # 绘制条形图
            bars = ax.barh(y_pos + y_offset, differences, height,
                          color=comparison_colors[method]['color'],
                          alpha=0.8,
                          edgecolor='black',
                          linewidth=0.8,
                          label=comparison_colors[method]['label'])
            
            # 添加数值标签
            for j, (bar, diff, sub_error) in enumerate(zip(bars, differences, sorted_sub_errors)):
                if abs(diff) > 0.001:
                    label_x = bar.get_width() + (0.005 if diff >= 0 else -0.005)
                    ha = 'left' if diff >= 0 else 'right'
                    ax.text(label_x, bar.get_y() + bar.get_height()/2,
                           f'{diff:+.3f}', ha=ha, va='center', fontsize=9, fontweight='bold')
        
        # 添加baseline参考线
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        
        # 设置标签和标题
        category_display = category
        suffixes_to_remove = ['-Related Errors', ' Errors', '-related', '_error', '_errors', '-Related']
        for suffix in suffixes_to_remove:
            if category_display.endswith(suffix):
                category_display = category_display[:-len(suffix)]
                break
        
        baseline_short = baseline_method.split('(')[0].strip()
        ax.set_title(f'{category_display} - Detailed Comparison vs {baseline_short}', 
                    fontsize=14, fontweight='bold')
        
        if idx == len(worst_categories) - 1:
            ax.set_xlabel('Difference from Baseline', fontsize=12, fontweight='bold')
        
        # 应用名称映射（如果有）
        display_sub_errors = [name_mapping.get(err, err) for err in sorted_sub_errors]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_sub_errors, fontsize=10)
        
        # 设置x轴范围
        if len(differences) > 0:
            all_diffs = [d for d in differences if d != 0]
            if all_diffs:
                max_abs_diff = max([abs(d) for d in all_diffs] + [0.05])
                ax.set_xlim(-max_abs_diff * 1.2, max_abs_diff * 1.2)
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        if idx == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=10, loc='upper right')
    
    baseline_short = baseline_method.split('(')[0].strip()
    plt.suptitle(f'Detailed Error Type Comparison Relative to {baseline_short}\n(Worst Performing Categories)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def create_summary_statistics(comp_df, baseline_method, comparison_colors):
    """创建统计摘要图 - 动态适配"""
    methods = comp_df['Method'].unique()
    n_methods = len(methods)
    
    # 动态调整子图布局
    if n_methods == 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 改进/退化统计
    improvement_stats = comp_df.groupby('Method').apply(
        lambda x: pd.Series({
            'Improved': (x['Difference'] > 0).sum(),
            'Degraded': (x['Difference'] < 0).sum(),
            'Unchanged': (x['Difference'] == 0).sum(),
            'Avg_Improvement': x[x['Difference'] > 0]['Difference'].mean() if (x['Difference'] > 0).any() else 0,
            'Avg_Degradation': x[x['Difference'] < 0]['Difference'].mean() if (x['Difference'] < 0).any() else 0
        })
    ).fillna(0)
    
    x = np.arange(len(methods))
    width = 0.25
    
    ax1.bar(x - width, improvement_stats['Improved'], width, label='Improved', 
           color='green', alpha=0.7)
    ax1.bar(x, improvement_stats['Degraded'], width, label='Degraded', 
           color='red', alpha=0.7)
    ax1.bar(x + width, improvement_stats['Unchanged'], width, label='Unchanged', 
           color='gray', alpha=0.7)
    
    ax1.set_title('Improvement/Degradation Count', fontweight='bold')
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Number of Categories')
    ax1.set_xticks(x)
    ax1.set_xticklabels([comparison_colors[m]['label'] for m in methods], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 平均改进/退化幅度
    ax2.bar(x - width/2, improvement_stats['Avg_Improvement'], width, 
           label='Avg Improvement', color='green', alpha=0.7)
    ax2.bar(x + width/2, improvement_stats['Avg_Degradation'], width, 
           label='Avg Degradation', color='red', alpha=0.7)
    
    ax2.set_title('Average Improvement/Degradation Magnitude', fontweight='bold')
    ax2.set_xlabel('Methods')
    ax2.set_ylabel('Average Difference')
    ax2.set_xticks(x)
    ax2.set_xticklabels([comparison_colors[m]['label'] for m in methods], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # 3. 分布直方图
    for method in methods:
        method_diffs = comp_df[comp_df['Method'] == method]['Difference']
        ax3.hist(method_diffs, bins=10, alpha=0.6, 
                label=comparison_colors[method]['label'],
                color=comparison_colors[method]['color'])
    
    ax3.set_title('Distribution of Differences', fontweight='bold')
    ax3.set_xlabel('Difference from Baseline')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
    # 4. 累积改进效果
    for method in methods:
        method_data = comp_df[comp_df['Method'] == method].sort_values('Difference')
        cumulative = method_data['Difference'].cumsum()
        ax4.plot(range(len(cumulative)), cumulative, 
                marker='o', label=comparison_colors[method]['label'], linewidth=2)
    
    ax4.set_title('Cumulative Improvement Effect', fontweight='bold')
    ax4.set_xlabel('Category Rank (sorted by difference)')
    ax4.set_ylabel('Cumulative Difference')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.8)
    
    baseline_short = baseline_method.split('(')[0].strip()
    plt.suptitle(f'Statistical Summary of {baseline_short} Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return improvement_stats

# ==== 主执行函数 ====
def main():
    """主函数：生成动态的baseline对比图表"""
    print("Loading and processing data...")
    (df, sub_error_to_category, ordered_categories, ordered_sub_errors, 
     baseline_method, category_colors, comparison_colors, metric_key) = load_and_process_data()
    
    print(f"Detected baseline method: {baseline_method}")
    print(f"Detected metric: {metric_key}")
    print(f"Found {len(ordered_categories)} categories and {len(FILES)} methods")
    
    print("Creating baseline comparison chart...")
    comp_df, ordered_cats, baseline_sorted = create_baseline_comparison_chart(
        df, ordered_categories, baseline_method, category_colors, comparison_colors, metric_key)
    
    print("Creating detailed baseline comparison...")
    create_detailed_baseline_comparison(df, sub_error_to_category, ordered_cats, baseline_sorted,
                                      baseline_method, comparison_colors)
    
    print("Creating summary statistics...")
    improvement_stats = create_summary_statistics(comp_df, baseline_method, comparison_colors)
    
    # 打印统计摘要
    baseline_short = baseline_method.split('(')[0].strip()
    print(f"\n=== {baseline_short} 对比统计摘要 ===")
    print("大类baseline性能排序 (从低到高):")
    for i, (category, score) in enumerate(baseline_sorted.items(), 1):
        cat_display = category
        suffixes_to_remove = ['-Related Errors', ' Errors', '-related', '_error', '_errors']
        for suffix in suffixes_to_remove:
            if cat_display.endswith(suffix):
                cat_display = cat_display[:-len(suffix)]
                break
        print(f"{i}. {cat_display}: {score:.3f}")
    
    print("\n各方法改进/退化统计:")
    for method in improvement_stats.index:
        method_label = comparison_colors[method]['label']
        stats = improvement_stats.loc[method]
        print(f"{method_label}:")
        print(f"  - 改进类别: {int(stats['Improved'])} 个")
        print(f"  - 退化类别: {int(stats['Degraded'])} 个") 
        print(f"  - 无变化类别: {int(stats['Unchanged'])} 个")
        print(f"  - 平均改进幅度: {stats['Avg_Improvement']:.4f}")
        print(f"  - 平均退化幅度: {stats['Avg_Degradation']:.4f}")

if __name__ == "__main__":
    main()