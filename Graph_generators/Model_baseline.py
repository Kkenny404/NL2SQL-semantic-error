import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import seaborn as sns
from matplotlib.patches import Polygon

# ==== 数据集配置 ====
DATA_FILES = {
    "GPT4.1": "results/SSS/eval_results/erros/Basline_error.json",
    "GPT-4o": "results/SSS/eval_results/erros/4o_error.json",
    "Gemini-2.5-Flash": "results/SSS/eval_results/erros/gemini_error.json", 
}

# 模型颜色配置 - 学术风格
MODEL_COLORS = {
    "GPT4.1": {'color': '#2E86AB', 'alpha': 0.25, 'linewidth': 2.5},  # 深蓝
    "GPT-4o": {'color': "#321B99", 'alpha': 0.25, 'linewidth': 2.5},  # 深青色
    "Gemini-2.5-Flash": {'color': '#A23B72', 'alpha': 0.25, 'linewidth': 2.5},   # 深紫红
    # "Claude-Sonnect-4": {'color': '#F18F01', 'alpha': 0.25, 'linewidth': 2.5},    # 深橙
}


def load_and_process_multi_model_data():
    """加载和处理多模型数据"""
    # 从第一个文件获取数据结构
    first_file = list(DATA_FILES.values())[0]
    with open(first_file) as f:
        sample_data = json.load(f)
    
    # 动态识别数据结构
    skip_keys = {"average_recall", "avg_recall", "total", "summary"}
    
    # 建立错误类型映射
    categories = []
    category_mapping = {}
    
    def find_metric_key(metrics_dict):
        """动态找到性能指标键"""
        possible_keys = ["recall", "accuracy", "f1", "score", "performance", "value"]
        for key in possible_keys:
            if key in metrics_dict:
                return key
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and key not in skip_keys:
                return key
        return None
    
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
                    metric_key = find_metric_key(metrics)
                    if metric_key and metrics.get(metric_key) is not None:
                        sub_values.append(metrics[metric_key])
                else:
                    sub_values.append(metrics)
            category_mapping[big_category] = sub_values
        else:
            # 平面结构
            category_mapping[big_category] = [content]
    
    # 读取所有模型数据
    all_model_data = {}
    metric_key = None
    
    for model_name, file_path in DATA_FILES.items():
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
                            if metric_key is None:
                                metric_key = find_metric_key(metrics)
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
        # 移除常见后缀
        clean_name = category
        suffixes_to_remove = ['-Related Errors', ' Errors', '-related', '_error', '_errors']
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
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # 设置角度和标签
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], 
                      [clean_names[cat] for cat in categories])
    
    # 设置固定刻度范围 0.5-1.0
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
            'color': '#333333', 'alpha': 0.25, 'linewidth': 2
        })
        
        # 绘制雷达图线条
        ax.plot(angles, values, 'o-', linewidth=color_config['linewidth'], 
                label=model_name, color=color_config['color'], markersize=6)
        
        # 填充区域
        ax.fill(angles, values, alpha=color_config['alpha'], color=color_config['color'])
    
    # 设置网格和标题
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Multi-Model Performance Comparison\n({metric_key or "Performance"} across Error Categories)', 
                size=16, fontweight='bold', pad=30)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return all_model_data

def create_detailed_radar_comparison(all_model_data, categories, metric_key="Performance"):
    """创建详细的多雷达图对比 - 每个模型一个子图"""
    num_models = len(all_model_data)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 6), 
                            subplot_kw=dict(projection='polar'))
    
    if num_models == 1:
        axes = [axes]
    
    # 清理类别名称
    clean_names = clean_category_names(categories)
    
    # 设置雷达图参数
    num_categories = len(categories)
    angles = [n / float(num_categories) * 2 * pi for n in range(num_categories)]
    angles += angles[:1]
    
    # 设置固定刻度范围 0.5-1.0
    y_min = 0.5
    y_max = 1.0
    
    for idx, (model_name, model_data) in enumerate(all_model_data.items()):
        ax = axes[idx]
        
        # 设置角度和标签
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], 
                          [clean_names[cat] for cat in categories])
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
                color=color_config['color'], markersize=8)
        ax.fill(angles, values, alpha=color_config['alpha'], color=color_config['color'])
        
        # 添加数值标签
        for angle, value in zip(angles[:-1], values[:-1]):
            ax.text(angle, value + (y_max - y_min) * 0.05, f'{value:.3f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_title(f'{model_name}', size=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Individual Model Performance Profiles\n({metric_key or "Performance"} by Error Category)', 
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
    plt.figure(figsize=(12, 6))
    
    # 使用自定义颜色映射
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
    
    # 绘制热图
    ax = sns.heatmap(data_matrix, 
                     xticklabels=[clean_names[cat] for cat in categories],
                     yticklabels=model_names,
                     annot=True, fmt='.3f', 
                     cmap=cmap, center=np.mean(data_matrix),
                     cbar_kws={'label': f'{metric_key or "Performance"}'})
    
    plt.title(f'Model Performance Heatmap\n({metric_key or "Performance"} across Error Categories)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Error Categories', fontsize=12, fontweight='bold')
    plt.ylabel('Models', fontsize=12, fontweight='bold')
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
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
    
    plt.figure(figsize=(12, 6))
    
    # 创建排名热图（数值越小颜色越深，表示排名越好）
    sns.heatmap(rank_matrix, annot=True, fmt='d', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (1=Best)'})
    
    plt.title(f'Model Ranking Matrix\n(1=Best Performance per Category)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Error Categories', fontsize=12, fontweight='bold')
    plt.ylabel('Models', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return ranking_df

def print_comprehensive_analysis(all_model_data, categories, metric_key="Performance"):
    """打印综合分析报告"""
    print(f"\n{'='*80}")
    print(f"               多模型性能对比分析报告")
    print(f"{'='*80}")
    
    models = list(all_model_data.keys())
    
    # 1. 整体性能统计
    print(f"\n📊 整体性能统计:")
    print(f"{'模型名称':<20} {'平均性能':<12} {'最高性能':<12} {'最低性能':<12} {'标准差':<10}")
    print("-" * 70)
    
    for model in models:
        values = list(all_model_data[model].values())
        avg_perf = np.mean(values)
        max_perf = np.max(values)
        min_perf = np.min(values)
        std_perf = np.std(values)
        
        print(f"{model:<20} {avg_perf:<12.4f} {max_perf:<12.4f} {min_perf:<12.4f} {std_perf:<10.4f}")
    
    # 2. 各类别最佳模型
    print(f"\n🏆 各错误类别最佳模型:")
    print(f"{'错误类别':<25} {'最佳模型':<20} {'性能值':<10}")
    print("-" * 60)
    
    clean_names = clean_category_names(categories)
    
    for category in categories:
        best_model = max(models, key=lambda m: all_model_data[m][category])
        best_score = all_model_data[best_model][category]
        print(f"{clean_names[category]:<25} {best_model:<20} {best_score:<10.4f}")
    
    # 3. 模型优势分析
    print(f"\n💪 模型优势分析:")
    for model in models:
        # 计算该模型在多少个类别上表现最佳
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
    
    # 4. 改进建议
    print(f"\n💡 模型改进建议:")
    for model in models:
        model_data = all_model_data[model]
        # 找出该模型表现最差的3个类别
        worst_categories = sorted(categories, key=lambda c: model_data[c])[:3]
        
        print(f"\n{model} 需要重点改进的领域:")
        for i, category in enumerate(worst_categories, 1):
            score = model_data[category]
            # 找出在该类别表现最好的模型作为参考
            best_model_in_cat = max(models, key=lambda m: all_model_data[m][category])
            best_score_in_cat = all_model_data[best_model_in_cat][category]
            gap = best_score_in_cat - score
            
            print(f"  {i}. {clean_names[category]}: {score:.4f} "
                  f"(落后{best_model_in_cat} {gap:.4f})")

def main():
    """主函数"""
    print("Loading multi-model data...")
    all_model_data, categories, metric_key = load_and_process_multi_model_data()
    
    models = list(all_model_data.keys())
    print(f"Found {len(models)} models: {', '.join(models)}")
    print(f"Analyzing {len(categories)} error categories")
    print(f"Performance metric: {metric_key or 'Performance'}")
    
    print("\n1. Creating radar chart comparison...")
    create_radar_chart(all_model_data, categories, metric_key)
    
    print("\n2. Creating detailed radar comparison...")
    create_detailed_radar_comparison(all_model_data, categories, metric_key)
    
    print("\n3. Creating performance heatmap...")
    data_matrix = create_performance_heatmap(all_model_data, categories, metric_key)
    
    print("\n4. Creating ranking analysis...")
    ranking_df = create_ranking_analysis(all_model_data, categories, metric_key)
    
    print("\n5. Generating comprehensive analysis...")
    print_comprehensive_analysis(all_model_data, categories, metric_key)

if __name__ == "__main__":
    main()