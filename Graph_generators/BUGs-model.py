import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from math import pi

# ==== 配置 ====
ALL_ERRORS_FILE = "Spider/error_explaination.json"
FILES = {
    "GPT4.1": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
    "GPT-4o": "results/eval_result/errors/4o.json",
    "Gemini-2.5-Flash": "results/eval_result/errors/gemini.json",
    "Claude-Sonnect-4": "results/eval_result/errors/claude.json",
}

MODEL_COLORS = {
    "GPT4.1": {'color': '#2E86AB', 'alpha': 0.25, 'linewidth': 2.5},  # 深蓝
    "GPT-4o": {'color': "#321B99", 'alpha': 0.25, 'linewidth': 2.5},  # 深青色
    "Gemini-2.5-Flash": {'color': '#A23B72', 'alpha': 0.25, 'linewidth': 2.5},   # 深紫红
    "Claude-Sonnect-4": {'color': '#F18F01', 'alpha': 0.25, 'linewidth': 2.5},    # 深橙
}


# # ==== 数据集配置 ====
# FILES = {
#     "GPT4.1": "results/SSS/eval_results/erros/Basline_error.json",
#     "GPT-4o": "results/SSS/eval_results/erros/4o_error.json",
#     "Gemini-2.5-Flash": "results/SSS/eval_results/erros/gemini_error.json", 
# }

# # 模型颜色配置 - 学术风格
# MODEL_COLORS = {
#     "GPT4.1": {'color': '#2E86AB', 'alpha': 0.25, 'linewidth': 2.5},  # 深蓝
#     "GPT-4o": {'color': "#321B99", 'alpha': 0.25, 'linewidth': 2.5},  # 深青色
#     "Gemini-2.5-Flash": {'color': '#A23B72', 'alpha': 0.25, 'linewidth': 2.5},   # 深紫红
#     # "Claude-Sonnect-4": {'color': '#F18F01', 'alpha': 0.25, 'linewidth': 2.5},    # 深橙
# }


# ==== 名称映射（用于美化显示） ====
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

def identify_baseline_method(methods):
    """智能识别baseline方法"""
    for method in methods:
        if any(keyword in method.lower() for keyword in ['baseline', 'base', 'reference', 'simple']):
            return method
    return methods[0]

def find_metric_key(sample_metrics):
    """动态找到性能指标键"""
    possible_keys = ["recall", "accuracy", "f1", "f1_score", "score", "performance", "value", "precision"]
    skip_keys = {"average_recall", "avg_recall", "total", "summary", "count"}
    
    if isinstance(sample_metrics, dict):
        for key in possible_keys:
            if key in sample_metrics and key not in skip_keys:
                return key
        
        for key, value in sample_metrics.items():
            if isinstance(value, (int, float)) and key not in skip_keys:
                return key
    
    return "recall"

def load_and_process_multi_model_data():
    """加载和处理多模型数据"""
    # 从第一个文件获取数据结构
    sample_file = list(FILES.values())[0]
    with open(sample_file) as f:
        sample_data = json.load(f)
    
    # 动态识别跳过的键
    skip_keys = {"average_recall", "avg_recall", "total", "summary", "count", "overall"}
    
    # 建立错误类型层次结构
    categories = []
    category_mapping = {}
    metric_key = None
    
    # 分析数据结构
    for big_category, content in sample_data.items():
        if big_category in skip_keys:
            continue
        
        categories.append(big_category)
        
        if isinstance(content, dict):
            # 嵌套结构 - 计算大类的平均值
            sub_values = []
            for sub_error, metrics in content.items():
                if sub_error in skip_keys:
                    continue
                if isinstance(metrics, dict):
                    if metric_key is None:
                        metric_key = find_metric_key(metrics)
                    if metric_key and metrics.get(metric_key) is not None:
                        sub_values.append(metrics[metric_key])
                else:
                    sub_values.append(metrics)
            category_mapping[big_category] = sub_values
        else:
            # 平面结构
            category_mapping[big_category] = [content]
    
    if metric_key is None:
        metric_key = "recall"
    
    # 读取所有模型数据并整合
    all_model_data = {}
    
    for model_name, file_path in FILES.items():
        with open(file_path) as f:
            model_data = json.load(f)
        
        model_performance = {}
        
        for category in categories:
            if category in model_data:
                content = model_data[category]
                
                if isinstance(content, dict):
                    # 嵌套结构 - 计算平均值
                    values = []
                    for sub_error, metrics in content.items():
                        if sub_error in skip_keys:
                            continue
                        if isinstance(metrics, dict):
                            if metric_key and metrics.get(metric_key) is not None:
                                values.append(metrics[metric_key])
                        else:
                            values.append(metrics)
                    
                    if values:
                        model_performance[category] = np.mean(values)
                    else:
                        model_performance[category] = 0
                else:
                    # 平面结构
                    model_performance[category] = content
            else:
                model_performance[category] = 0
        
        all_model_data[model_name] = model_performance
    
    return all_model_data, categories, metric_key

def clean_category_names(categories):
    """清理类别名称，使其更适合显示"""
    cleaned = {}
    for category in categories:
        # 首先应用名称映射
        clean_name = name_mapping.get(category, category)
        
        # 移除常见后缀
        suffixes_to_remove = ['-Related Errors', ' Errors', '-related', '_error', '_errors', '-Related']
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # 进一步简化长名称
        if len(clean_name) > 15:
            words = clean_name.split()
            if len(words) > 2:
                clean_name = ' '.join(words[:2]) + '...'
            elif len(words) == 2:
                clean_name = words[0][:8] + ' ' + words[1][:8]
            else:
                clean_name = clean_name[:15] + '...'
        
        cleaned[category] = clean_name
    
    return cleaned

def create_radar_chart(all_model_data, categories, metric_key="Performance"):
    """创建多模型雷达图对比"""
    # 清理类别名称
    clean_names = clean_category_names(categories)
    
    # 设置雷达图参数
    num_categories = len(categories)
    angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
    angles += angles[:1]  # 闭合雷达图
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
    
    # 设置角度和标签
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], 
                      [clean_names[cat] for cat in categories], fontsize=12, fontweight='bold')
    
    # 动态设置刻度范围
    all_values = []
    for model_data in all_model_data.values():
        all_values.extend(model_data.values())
    
    min_val = min(all_values) if all_values else 0
    max_val = max(all_values) if all_values else 1
    
    # 设置合适的刻度范围
    range_val = max_val - min_val
    y_min = max(0, min_val - range_val * 0.1)
    y_max = max_val + range_val * 0.1

    y_min = 0.5
    y_max = 1.0
    
    
    ax.set_ylim(y_min, y_max)
    
    # 绘制每个模型
    for model_name, model_data in all_model_data.items():
        # 获取各类别的性能值
        values = [model_data[cat] for cat in categories]
        values += values[:1]  # 闭合雷达图
        
        # 获取模型颜色配置
        color_config = MODEL_COLORS.get(model_name, {
            'color': '#333333', 'alpha': 0.25, 'linewidth': 2.5
        })
        
        # 绘制雷达图线条
        ax.plot(angles, values, 'o-', linewidth=color_config['linewidth'], 
                label=model_name, color=color_config['color'], markersize=8)
        
        # 填充区域
        ax.fill(angles, values, alpha=color_config['alpha'], color=color_config['color'])
    
    # 设置网格和标题
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Cross-Model Capability Comparison\n(Negative Recall by Error Category on SSS Dataset)',
                size=16, fontweight='bold', pad=30)
    
    # 设置刻度标签
    ax.tick_params(axis='y', labelsize=10)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=13)
    
    plt.tight_layout()
    plt.show()
    
    return all_model_data

def create_detailed_radar_comparison(all_model_data, categories, metric_key="Performance"):
    """创建详细的多雷达图对比 - 每个模型一个子图"""
    num_models = len(all_model_data)
    fig, axes = plt.subplots(1, num_models, figsize=(7*num_models, 7), 
                            subplot_kw=dict(projection='polar'))
    
    if num_models == 1:
        axes = [axes]
    
    # 清理类别名称
    clean_names = clean_category_names(categories)
    
    # 设置雷达图参数
    num_categories = len(categories)
    angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
    angles += angles[:1]
    
    # 获取全局刻度范围
    all_values = []
    for model_data in all_model_data.values():
        all_values.extend(model_data.values())
    min_val = min(all_values) if all_values else 0
    max_val = max(all_values) if all_values else 1
    range_val = max_val - min_val
    y_min = max(0, min_val - range_val * 0.1)
    y_max = max_val + range_val * 0.1
    
    for idx, (model_name, model_data) in enumerate(all_model_data.items()):
        ax = axes[idx]
        
        # 设置角度和标签
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], 
                          [clean_names[cat] for cat in categories], fontsize=10)
        ax.set_ylim(y_min, y_max)
        
        # 获取数据
        values = [model_data[cat] for cat in categories]
        values += values[:1]
        
        # 获取颜色配置
        color_config = MODEL_COLORS.get(model_name, {
            'color': '#333333', 'alpha': 0.3, 'linewidth': 2.5
        })
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=color_config['linewidth'], 
                color=color_config['color'], markersize=10)
        ax.fill(angles, values, alpha=color_config['alpha'], color=color_config['color'])
        
        # 添加数值标签
        for angle, value in zip(angles[:-1], values[:-1]):
            ax.text(angle, value + (y_max - y_min) * 0.06, f'{value:.3f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{model_name}', size=14, fontweight='bold', pad=25)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=9)
    
    plt.suptitle(f'Individual Model Performance Profiles\n({metric_key.title()} by Error Category - NL2SQL-BUGs Dataset)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def create_performance_heatmap(all_model_data, categories, metric_key="Performance"):
    """创建性能热图矩阵"""
    # 准备数据
    clean_names = clean_category_names(categories)
    
    # 创建数据矩阵
    data_matrix = []
    model_names = list(all_model_data.keys())
    
    for model_name in model_names:
        model_values = [all_model_data[model_name][cat] for cat in categories]
        data_matrix.append(model_values)
    
    data_matrix = np.array(data_matrix)
    
    # 创建热图
    plt.figure(figsize=(14, 8))
    
    # 使用自定义颜色映射
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
    
    # 绘制热图
    ax = sns.heatmap(data_matrix, 
                     xticklabels=[clean_names[cat] for cat in categories],
                     yticklabels=model_names,
                     annot=True, fmt='.3f', 
                     cmap=cmap, center=np.mean(data_matrix),
                     cbar_kws={'label': f'{metric_key.title()}'})
    
    plt.title(f'Model Performance Heatmap\n({metric_key.title()} across Error Categories - NL2SQL-BUGs Dataset)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Error Categories', fontsize=13, fontweight='bold')
    plt.ylabel('Models', fontsize=13, fontweight='bold')
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return data_matrix

def create_ranking_analysis(all_model_data, categories, metric_key="Performance"):
    """创建排名分析图"""
    # 计算每个类别上的模型排名
    ranking_data = []
    clean_names = clean_category_names(categories)
    
    for category in categories:
        # 获取所有模型在该类别上的性能
        category_scores = [(model, data[category]) for model, data in all_model_data.items()]
        # 按性能排序（降序）
        category_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 记录排名
        for rank, (model, score) in enumerate(category_scores, 1):
            ranking_data.append({
                'Category': clean_names[category],
                'Model': model,
                'Performance': score,
                'Rank': rank
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # 创建排名矩阵热图
    rank_matrix = ranking_df.pivot(index='Model', columns='Category', values='Rank')
    
    plt.figure(figsize=(14, 8))
    
    # 创建排名热图（数值越小颜色越深，表示排名越好）
    sns.heatmap(rank_matrix, annot=True, fmt='d', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (1=Best)'})
    
    plt.title(f'Model Ranking Matrix\n(1=Best Performance per Category - NL2SQL-BUGs Dataset)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Error Categories', fontsize=13, fontweight='bold')
    plt.ylabel('Models', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return ranking_df

def create_comprehensive_comparison_chart(all_model_data, categories, metric_key="Performance"):
    """创建综合对比图表"""
    # 基于原脚本的基线对比风格，但适配多模型
    baseline_method = identify_baseline_method(list(all_model_data.keys()))
    comparison_methods = [m for m in all_model_data.keys() if m != baseline_method]
    
    if not comparison_methods:
        print("No comparison methods found for comprehensive chart.")
        return
    
    # 计算相对于baseline的差异
    baseline_data = all_model_data[baseline_method]
    comparison_data = []
    
    for method in comparison_methods:
        method_data = all_model_data[method]
        for category in categories:
            baseline_val = baseline_data[category]
            method_val = method_data[category]
            difference = method_val - baseline_val
            percentage_diff = (difference / baseline_val * 100) if baseline_val > 0 else 0
            
            comparison_data.append({
                'Method': method,
                'Category': category,
                'Difference': difference,
                'Percentage_Diff': percentage_diff,
                'Baseline_Value': baseline_val,
                'Method_Value': method_val
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # 按baseline性能排序类别
    baseline_sorted = pd.Series(baseline_data).sort_values(ascending=True)
    ordered_cats = baseline_sorted.index.tolist()
    
    # 清理类别名称
    clean_names = clean_category_names(categories)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 设置y轴位置
    y_pos = np.arange(len(ordered_cats))
    height = 0.8 / len(comparison_methods) if len(comparison_methods) > 1 else 0.6
    
    # 动态颜色配置
    comparison_colors = {}
    colors = ['#AC6ECB', '#36C5F0', '#4682B4', '#2E8B57', '#FF8C00']
    for i, method in enumerate(comparison_methods):
        comparison_colors[method] = {
            'color': colors[i % len(colors)],
            'label': method
        }
    
    # 绘制对比条形图
    for i, method in enumerate(comparison_methods):
        method_data = comp_df[comp_df['Method'] == method]
        
        differences = []
        for cat in ordered_cats:
            cat_data = method_data[method_data['Category'] == cat]
            if not cat_data.empty:
                differences.append(cat_data['Difference'].iloc[0])
            else:
                differences.append(0)
        
        # 计算y位置偏移
        y_offset = (i - (len(comparison_methods) - 1) / 2) * height if len(comparison_methods) > 1 else 0
        
        # 绘制水平条形图
        bars = ax.barh(y_pos + y_offset, differences, height,
                      label=comparison_colors[method]['label'],
                      color=comparison_colors[method]['color'],
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=0.8)
        
        # 添加数值标签
        for j, (bar, diff) in enumerate(zip(bars, differences)):
            if abs(diff) > 0.001:
                label_x = bar.get_width() + (0.01 if diff >= 0 else -0.01)
                ha = 'left' if diff >= 0 else 'right'
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       f'{diff:+.3f}', ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # 添加baseline参考线
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Baseline')
    
    # 设置标签和标题
    ax.set_xlabel(f'Difference from Baseline (Negative {metric_key.title()})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error Categories', fontsize=14, fontweight='bold')
    ax.set_title(f'Prompting Methods - Performance Comparison Relative to Baseline\n(NL2SQL-BUGs Dataset)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 设置y轴标签
    ax.set_yticks(y_pos)
    y_labels = [clean_names[cat] for cat in ordered_cats]
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    
    # 设置x轴
    if len(comp_df) > 0:
        max_abs_diff = max([abs(d) for d in comp_df['Difference']] + [0.1])
        ax.set_xlim(-max_abs_diff * 1.3, max_abs_diff * 1.3)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 添加图例
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def print_comprehensive_analysis(all_model_data, categories, metric_key="Performance"):
    """打印综合分析报告"""
    print(f"\n{'='*80}")
    print(f"            NL2SQL-BUGs数据集多模型性能对比分析报告")
    print(f"{'='*80}")
    
    models = list(all_model_data.keys())
    
    # 1. 整体性能统计
    print(f"\n📊 整体性能统计:")
    print(f"{'模型名称':<25} {'平均性能':<12} {'最高性能':<12} {'最低性能':<12} {'标准差':<10}")
    print("-" * 75)
    
    for model in models:
        values = list(all_model_data[model].values())
        avg_perf = np.mean(values)
        max_perf = np.max(values)
        min_perf = np.min(values)
        std_perf = np.std(values)
        
        print(f"{model:<25} {avg_perf:<12.4f} {max_perf:<12.4f} {min_perf:<12.4f} {std_perf:<10.4f}")
    
    # 2. 各类别最佳模型
    print(f"\n🏆 各错误类别最佳模型:")
    print(f"{'错误类别':<30} {'最佳模型':<25} {'性能值':<10}")
    print("-" * 70)
    
    clean_names = clean_category_names(categories)
    
    for category in categories:
        best_model = max(models, key=lambda m: all_model_data[m][category])
        best_score = all_model_data[best_model][category]
        print(f"{clean_names[category]:<30} {best_model:<25} {best_score:<10.4f}")
    
    # 3. 模型优势分析
    print(f"\n💪 模型优势分析:")
    for model in models:
        best_count = 0
        best_categories = []
        
        for category in categories:
            if max(models, key=lambda m: all_model_data[m][category]) == model:
                best_count += 1
                best_categories.append(clean_names[category])
        
        print(f"\n{model}:")
        print(f"  - 最佳表现类别数: {best_count}/{len(categories)} ({best_count/len(categories)*100:.1f}%)")
        if best_categories:
            print(f"  - 优势领域: {', '.join(best_categories)}")
        
        # 计算相对于其他模型的平均优势
        advantages = []
        for other_model in models:
            if other_model != model:
                for category in categories:
                    diff = all_model_data[model][category] - all_model_data[other_model][category]
                    advantages.append(diff)
        
        if advantages:
            avg_advantage = np.mean(advantages)
            print(f"  - 平均性能优势: {avg_advantage:+.4f}")

def main():
    """主函数"""
    print("Loading multi-model data for NL2SQL-BUGs dataset...")
    all_model_data, categories, metric_key = load_and_process_multi_model_data()
    
    models = list(all_model_data.keys())
    print(f"Found {len(models)} models: {', '.join(models)}")
    print(f"Analyzing {len(categories)} error categories")
    print(f"Performance metric: {metric_key}")
    
    print("\n1. Creating comprehensive radar chart comparison...")
    create_radar_chart(all_model_data, categories, metric_key)
    
    print("\n2. Creating detailed radar comparison...")
    create_detailed_radar_comparison(all_model_data, categories, metric_key)
    
    print("\n3. Creating performance heatmap...")
    data_matrix = create_performance_heatmap(all_model_data, categories, metric_key)
    
    print("\n4. Creating ranking analysis...")
    ranking_df = create_ranking_analysis(all_model_data, categories, metric_key)
    
    print("\n5. Creating comprehensive comparison chart...")
    create_comprehensive_comparison_chart(all_model_data, categories, metric_key)
    
    print("\n6. Generating comprehensive analysis...")
    print_comprehensive_analysis(all_model_data, categories, metric_key)

if __name__ == "__main__":
    main()