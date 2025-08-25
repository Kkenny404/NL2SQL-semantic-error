import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ==== 数据集配置 ====
ALL_ERRORS_FILE = "Spider/error_explaination.json"  # 或者对应的错误文件
DATA_FILES = {
    "Baseline": "results/SSS/eval_results/erros/Basline_error.json",
    "No Schema Provided": "results/SSS/eval_results/erros/No_schema_error.json",
    # "Original SQL Only (Baseline)": "results/SSS/eval_results/erros/Basline_error.json",
    # "No Schema Provided": "results/SSS/eval_results/erros/No_schema_error.json",
    # "With Extra Parsed SQL": "results/SSS/eval_results/erros/parsed_sql_error.json",
    # 可以添加更多方法
    # "Simple (Baseline)": "results/SSS/eval_results/erros/Basline_error.json",
    # "Chain-of-Thought": "results/SSS/eval_results/erros/CoT_error.json",
    # "Self-Reflection": "results/SSS/eval_results/erros/selfF_error.json",

}

# 学术风格颜色映射 - 动态生成
def generate_color_mapping(categories):
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
    
    color_mapping = {}
    for i, category in enumerate(categories):
        color_mapping[category] = colors[i % len(colors)]
    
    return color_mapping

# 对比方法的颜色 - 动态生成
def generate_comparison_colors(methods, baseline_method):
    """动态生成对比方法颜色"""
    comparison_colors = [
        '#36C5F0',  # 天蓝色
        "#AC6ECB",  # 深兰花紫
        '#2EB67D',  # 绿色
        '#ECB22E',  # 黄色
        '#E01E5A',  # 粉红色
        '#6F42C1',  # 紫色
    ]
    
    colors = {}
    color_idx = 0
    for method in methods:
        if method != baseline_method:
            colors[method] = {
                'color': comparison_colors[color_idx % len(comparison_colors)], 
                'label': method
            }
            color_idx += 1
    
    return colors

def load_and_process_data():
    """加载和处理错误数据 - 动态读取键值"""
    # 读取全集错误类型（如果需要）
    try:
        with open(ALL_ERRORS_FILE) as f:
            categories = json.load(f)
    except FileNotFoundError:
        categories = None
    
    # 从第一个文件获取数据结构
    first_file = list(DATA_FILES.values())[0]
    with open(first_file) as f:
        sample_data = json.load(f)
    
    # 动态识别数据结构
    skip_keys = {"average_recall", "avg_recall", "total", "summary"}  # 可能的统计键
    
    # 建立子错误到大类的映射
    sub_error_to_category = {}
    ordered_categories = []
    ordered_sub_errors = []
    
    for big_category, content in sample_data.items():
        if big_category in skip_keys:
            continue
            
        # 如果content是字典且包含子项
        if isinstance(content, dict):
            ordered_categories.append(big_category)
            for sub_error, metrics in content.items():
                if sub_error not in skip_keys:
                    sub_error_to_category[sub_error] = big_category
                    ordered_sub_errors.append(sub_error)
        # 如果content直接是数值，则big_category就是错误类型
        elif isinstance(content, (int, float)):
            ordered_categories.append(big_category)
            ordered_sub_errors.append(big_category)
            sub_error_to_category[big_category] = big_category
    
    # 动态识别性能指标键
    def find_metric_key(metrics_dict):
        """动态找到性能指标键"""
        possible_keys = ["recall", "accuracy", "f1", "score", "performance", "value"]
        for key in possible_keys:
            if key in metrics_dict:
                return key
        # 如果都没找到，返回第一个数值型键
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and key not in skip_keys:
                return key
        return None
    
    # 读取方法结果并整合
    data = []
    metric_key = None
    
    for method, path in DATA_FILES.items():
        with open(path) as f:
            res = json.load(f)
        
        for big_category, content in res.items():
            if big_category in skip_keys:
                continue
                
            if isinstance(content, dict):
                # 如果是嵌套结构
                for sub_error, metrics in content.items():
                    if sub_error in skip_keys:
                        continue
                    
                    if isinstance(metrics, dict):
                        # 动态找到性能指标
                        if metric_key is None:
                            metric_key = find_metric_key(metrics)
                        
                        metric_value = metrics.get(metric_key, None) if metric_key else None
                    else:
                        metric_value = metrics
                    
                    data.append({
                        "Method": method,
                        "Error Type": sub_error,
                        "Big Category": big_category,
                        "Metric": metric_value
                    })
            else:
                # 如果是平面结构
                data.append({
                    "Method": method,
                    "Error Type": big_category,
                    "Big Category": big_category,
                    "Metric": content
                })
    
    # 动态生成颜色映射
    category_colors = generate_color_mapping(ordered_categories)
    
    return pd.DataFrame(data), sub_error_to_category, ordered_categories, ordered_sub_errors, category_colors, metric_key

def create_baseline_comparison_chart(df, ordered_categories, category_colors, metric_key="Metric"):
    """创建baseline对比图 - 动态适配"""
    methods = df['Method'].unique().tolist()
    
    # 动态识别baseline方法（通常是第一个或包含'baseline'关键词的）
    baseline_method = None
    for method in methods:
        if 'baseline' in method.lower() or method == methods[0]:
            baseline_method = method
            break
    if baseline_method is None:
        baseline_method = methods[0]
    
    comparison_methods = [m for m in methods if m != baseline_method]
    comparison_colors = generate_comparison_colors(methods, baseline_method)
    
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
    
    # 简化大类名称
    category_display_names = {}
    for cat in ordered_cats:
        # 动态清理类别名称
        display_name = cat
        suffixes_to_remove = ['-Related Errors', ' Errors', '-related', '_error', '_errors']
        for suffix in suffixes_to_remove:
            if display_name.endswith(suffix):
                display_name = display_name[:-len(suffix)]
                break
        category_display_names[cat] = display_name
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
        y_offset = (i - (len(comparison_methods) - 1) / 2) * height
        
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
                label_text = f'{diff:+.3f}({pct_change:+.1f}%)'
                
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       label_text, ha=ha, va='center', fontsize=10, fontweight='bold')
    
    # 添加baseline参考线（x=0）
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, label=f'{baseline_method} Reference')
    
    # 设置标签和标题
    metric_display = metric_key.title() if metric_key else "Performance"
    ax.set_xlabel(f'Difference from Baseline (Negative {metric_display})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error Categories', fontsize=14, fontweight='bold')
    ax.set_title(f'Prompting Methods - Performance Comparison Relative to Baseline\n (SSS Dataset)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 设置y轴标签并着色
    ax.set_yticks(y_pos)
    y_labels = [category_display_names[cat] for cat in ordered_cats]
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    
    # 为y轴标签着色
    for i, (label, cat) in enumerate(zip(ax.get_yticklabels(), ordered_cats)):
        label.set_color(category_colors.get(cat, 'black'))
    
    # 设置x轴
    all_differences = [diff for method_data in [comp_df[comp_df['Method'] == method]['Difference'].tolist() 
                                               for method in comparison_methods] for diff in method_data]
    if all_differences:
        max_abs_diff = max([abs(d) for d in all_differences] + [0.1])
        ax.set_xlim(-max_abs_diff * 1.4, max_abs_diff * 1.4)
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
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)
    
    # 添加baseline值信息
    baseline_text = f"Baseline Performance:\n"
    for cat in ordered_cats:
        baseline_val = baseline_sorted[cat]
        cat_display = category_display_names[cat]
        baseline_text += f"{cat_display}: {baseline_val:.3f}\n"
    
    ax.text(0.02, 0.98, baseline_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return comp_df, ordered_cats, baseline_sorted, baseline_method, comparison_methods

def create_detailed_comparison(df, sub_error_to_category, ordered_cats, baseline_sorted, 
                             baseline_method, comparison_methods, comparison_colors, category_colors):
    """创建详细子错误类型对比图 - 动态适配"""
    # 选择表现最差的3个大类进行详细分析
    worst_categories = list(baseline_sorted.head(3).index)
    
    fig, axes = plt.subplots(len(worst_categories), 1, figsize=(14, 6*len(worst_categories)))
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
            y_offset = (i - (len(comparison_methods) - 1) / 2) * height
            
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
        for suffix in ['-Related Errors', ' Errors', '-related', '_error', '_errors']:
            if category_display.endswith(suffix):
                category_display = category_display[:-len(suffix)]
                break
                
        ax.set_title(f'{category_display}: Detailed Comparison vs {baseline_method}', 
                    fontsize=14, fontweight='bold', color=category_colors.get(category, 'black'))
        
        if idx == len(worst_categories) - 1:
            ax.set_xlabel(f'Difference from Baseline', fontsize=12, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_sub_errors, fontsize=10)
        
        # 设置x轴范围
        all_diffs = [d for diffs in [comp_sub_df[comp_sub_df['Method'] == method]['Difference'].tolist() 
                                   for method in comparison_methods] for d in diffs]
        if all_diffs:
            max_abs_diff = max([abs(d) for d in all_diffs if d != 0] + [0.05])
            ax.set_xlim(-max_abs_diff * 1.3, max_abs_diff * 1.3)
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        if idx == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=11, loc='upper right')
    
    plt.suptitle(f'Detailed Error Type Comparison vs {baseline_method}\n(Worst Performing Categories)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def create_summary_statistics(comp_df, comparison_methods, comparison_colors, baseline_method):
    """创建统计摘要 - 动态适配多方法"""
    n_methods = len(comparison_methods)
    n_cols = min(2, n_methods)
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_methods == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for method_idx, method in enumerate(comparison_methods):
        method_data = comp_df[comp_df['Method'] == method]
        
        # 确定子图位置
        if n_methods == 1:
            ax_idx = (0, 0)
        else:
            ax_idx = (method_idx // n_cols, method_idx % n_cols)
        
        if method_idx < len(axes.flat):
            ax = axes[ax_idx] if n_methods > 1 else axes[0, 0]
            
            # 改进/退化统计
            improved_count = (method_data['Difference'] > 0).sum()
            degraded_count = (method_data['Difference'] < 0).sum()
            unchanged_count = (method_data['Difference'] == 0).sum()
            
            counts = [improved_count, degraded_count, unchanged_count]
            labels = ['Improved', 'Degraded', 'Unchanged']
            colors = ['green', 'red', 'gray']
            
            bars = ax.bar(labels, counts, color=colors, alpha=0.7)
            ax.set_title(f'{method}: Improvement/Degradation Count', fontweight='bold')
            ax.set_ylabel('Number of Categories')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 分布直方图（如果有足够的子图位置）
        if method_idx + len(comparison_methods) < len(axes.flat):
            ax2_idx = ((method_idx + len(comparison_methods)) // n_cols, 
                       (method_idx + len(comparison_methods)) % n_cols)
            ax2 = axes[ax2_idx] if n_methods > 1 else axes[1, 0]
            
            ax2.hist(method_data['Difference'], bins=8, alpha=0.6, 
                    color=comparison_colors[method]['color'], edgecolor='black')
            ax2.set_title(f'{method}: Distribution of Differences', fontweight='bold')
            ax2.set_xlabel('Difference from Baseline')
            ax2.set_ylabel('Frequency')
            ax2.grid(axis='y', alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
    plt.suptitle(f'Statistical Summary of {baseline_method} Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 返回统计信息
    stats = {}
    for method in comparison_methods:
        method_data = comp_df[comp_df['Method'] == method]
        improvements = method_data[method_data['Difference'] > 0]['Difference']
        degradations = method_data[method_data['Difference'] < 0]['Difference']
        
        stats[method] = {
            'improved': (method_data['Difference'] > 0).sum(),
            'degraded': (method_data['Difference'] < 0).sum(),
            'unchanged': (method_data['Difference'] == 0).sum(),
            'avg_improvement': improvements.mean() if len(improvements) > 0 else 0,
            'avg_degradation': degradations.mean() if len(degradations) > 0 else 0
        }
    
    return stats

# ==== 主执行函数 ====
def main():
    """主函数：生成动态的baseline对比图表"""
    print("Loading and processing data...")
    df, sub_error_to_category, ordered_categories, ordered_sub_errors, category_colors, metric_key = load_and_process_data()
    
    print(f"Detected metric: {metric_key}")
    print(f"Found {len(ordered_categories)} categories and {len(DATA_FILES)} methods")
    
    print("Creating baseline comparison chart...")
    comp_df, ordered_cats, baseline_sorted, baseline_method, comparison_methods = create_baseline_comparison_chart(
        df, ordered_categories, category_colors, metric_key)
    
    print("Creating detailed comparison...")
    comparison_colors = generate_comparison_colors(df['Method'].unique().tolist(), baseline_method)
    create_detailed_comparison(df, sub_error_to_category, ordered_cats, baseline_sorted, 
                             baseline_method, comparison_methods, comparison_colors, category_colors)
    
    print("Creating summary statistics...")
    stats = create_summary_statistics(comp_df, comparison_methods, comparison_colors, baseline_method)
    
    # 打印统计摘要
    print(f"\n=== Baseline对比统计摘要 (Baseline: {baseline_method}) ===")
    print("大类baseline性能排序 (从低到高):")
    for i, (category, score) in enumerate(baseline_sorted.items(), 1):
        cat_display = category
        for suffix in ['-Related Errors', ' Errors', '-related', '_error', '_errors']:
            if cat_display.endswith(suffix):
                cat_display = cat_display[:-len(suffix)]
                break
        print(f"{i}. {cat_display}: {score:.3f}")
    
    print(f"\n各方法改进/退化统计:")
    for method, method_stats in stats.items():
        print(f"\n{method}:")
        print(f"  - 改进类别: {method_stats['improved']} 个")
        print(f"  - 退化类别: {method_stats['degraded']} 个")
        print(f"  - 无变化类别: {method_stats['unchanged']} 个")
        print(f"  - 平均改进幅度: {method_stats['avg_improvement']:.4f}")
        print(f"  - 平均退化幅度: {method_stats['avg_degradation']:.4f}")

if __name__ == "__main__":
    main()