import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ==== 配置 ====
DATA_FILE = "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json"  # 你的baseline数据文件
METHOD_NAME = "Baseline Method"  # 方法名称

def load_baseline_data():
    """加载单一方法数据"""
    with open(DATA_FILE) as f:
        data = json.load(f)
    
    # 动态识别数据结构
    skip_keys = {"average_recall", "avg_recall", "total", "summary"}
    
    # 建立错误类型层次结构
    categories = []
    sub_errors = []
    error_data = []
    
    # 识别性能指标键
    def find_metric_key(metrics_dict):
        possible_keys = ["recall", "accuracy", "f1", "score", "performance", "value"]
        for key in possible_keys:
            if key in metrics_dict:
                return key
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) and key not in skip_keys:
                return key
        return None
    
    metric_key = None
    
    for big_category, content in data.items():
        if big_category in skip_keys:
            continue
            
        categories.append(big_category)
        
        if isinstance(content, dict):
            # 嵌套结构 - 有子错误类型
            for sub_error, metrics in content.items():
                if sub_error in skip_keys:
                    continue
                
                if isinstance(metrics, dict):
                    if metric_key is None:
                        metric_key = find_metric_key(metrics)
                    metric_value = metrics.get(metric_key, None) if metric_key else None
                else:
                    metric_value = metrics
                
                error_data.append({
                    "Big_Category": big_category,
                    "Sub_Error": sub_error,
                    "Performance": metric_value,
                    "Error_Level": "Sub"
                })
                sub_errors.append(sub_error)
        else:
            # 平面结构 - 直接是性能值
            error_data.append({
                "Big_Category": big_category,
                "Sub_Error": big_category,
                "Performance": content,
                "Error_Level": "Main"
            })
    
    df = pd.DataFrame(error_data)
    return df, categories, metric_key

def create_performance_overview(df, categories, metric_key="Performance"):
    """创建性能概览图 - 显示所有错误类型的性能分布"""
    # 计算大类平均性能
    category_performance = df.groupby('Big_Category')['Performance'].mean().sort_values()
    
    # 颜色映射 - 性能越低颜色越红，越高越绿
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(category_performance)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：水平条形图显示各类别性能
    bars = ax1.barh(range(len(category_performance)), category_performance.values, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 添加性能数值标签
    for i, (bar, performance) in enumerate(zip(bars, category_performance.values)):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{performance:.3f}', ha='left', va='center', fontweight='bold')
    
    # 设置标签
    category_labels = [cat.replace('-Related Errors', '').replace(' Errors', '') 
                      for cat in category_performance.index]
    ax1.set_yticks(range(len(category_performance)))
    ax1.set_yticklabels(category_labels, fontsize=11)
    ax1.set_xlabel(f'{metric_key or "Performance"}', fontsize=12, fontweight='bold')
    ax1.set_title(f'{METHOD_NAME}: Performance by Error Category\n(NL2SQL-BUGs Dataset)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加平均线
    overall_avg = df['Performance'].mean()
    ax1.axvline(x=overall_avg, color='red', linestyle='--', linewidth=2, 
                label=f'Overall Average: {overall_avg:.3f}')
    ax1.legend()
    
    # 右图：性能分布直方图
    ax2.hist(df['Performance'], bins=15, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=1)
    ax2.axvline(x=overall_avg, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {overall_avg:.3f}')
    ax2.axvline(x=df['Performance'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {df["Performance"].median():.3f}')
    
    ax2.set_xlabel(f'{metric_key or "Performance"}', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'{METHOD_NAME}: Performance Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return category_performance

def create_detailed_breakdown(df, category_performance):
    """创建详细分解图 - 显示性能最差的类别的子错误详情"""
    # 选择性能最差的3个大类
    worst_categories = category_performance.head(3).index.tolist()
    
    fig, axes = plt.subplots(len(worst_categories), 1, figsize=(14, 5*len(worst_categories)))
    if len(worst_categories) == 1:
        axes = [axes]
    
    for idx, category in enumerate(worst_categories):
        ax = axes[idx]
        
        # 获取该大类的所有子错误
        category_data = df[df['Big_Category'] == category].copy()
        category_data = category_data.sort_values('Performance')
        
        # 创建颜色映射
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(category_data)))
        
        # 绘制条形图
        bars = ax.bar(range(len(category_data)), category_data['Performance'], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar, performance in zip(bars, category_data['Performance']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{performance:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 设置标签
        ax.set_xticks(range(len(category_data)))
        ax.set_xticklabels(category_data['Sub_Error'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Performance', fontsize=11, fontweight='bold')
        
        # 简化类别名称
        category_display = category.replace('-Related Errors', '').replace(' Errors', '')
        ax.set_title(f'{category_display}: Detailed Breakdown', fontsize=13, fontweight='bold')
        
        # 添加该类别的平均线
        category_avg = category_data['Performance'].mean()
        ax.axhline(y=category_avg, color='blue', linestyle='--', alpha=0.7,
                   label=f'Category Average: {category_avg:.3f}')
        
        # 添加全局平均线
        global_avg = df['Performance'].mean()
        ax.axhline(y=global_avg, color='red', linestyle='--', alpha=0.7,
                   label=f'Overall Average: {global_avg:.3f}')
        
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{METHOD_NAME}: Detailed Analysis of Worst Performing Categories', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def create_performance_matrix(df):
    """创建性能矩阵热图"""
    # 创建透视表
    if df['Error_Level'].nunique() > 1 and 'Sub' in df['Error_Level'].values:
        # 如果有子错误，创建大类-子错误矩阵
        pivot_data = df[df['Error_Level'] == 'Sub'].pivot_table(
            values='Performance', index='Big_Category', columns='Sub_Error', 
            aggfunc='mean', fill_value=0)
        
        plt.figure(figsize=(15, 8))
        mask = pivot_data == 0  # 遮盖空值
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                   mask=mask, cbar_kws={'label': 'Performance'}, linewidths=0.5)
        
        plt.title(f'{METHOD_NAME}: Performance Heatmap (Categories vs Sub-Errors)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Sub Error Types', fontsize=12, fontweight='bold')
        plt.ylabel('Main Categories', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

def create_performance_ranking(df, metric_key="Performance"):
    """创建性能排名图 - 展示所有错误类型的排名"""
    # 按性能排序所有错误类型
    df_sorted = df.sort_values('Performance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(df_sorted) * 0.4)))
    
    # 创建颜色映射：前25%红色，中间50%黄色，后25%绿色
    n_items = len(df_sorted)
    colors = []
    for i in range(n_items):
        if i < n_items * 0.25:
            colors.append('#FF6B6B')  # 红色 - 性能差
        elif i < n_items * 0.75:
            colors.append('#FFE66D')  # 黄色 - 中等性能  
        else:
            colors.append('#4ECDC4')  # 绿色 - 性能好
    
    # 绘制水平条形图
    bars = ax.barh(range(len(df_sorted)), df_sorted['Performance'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 添加排名和数值标签
    for i, (bar, performance, category, sub_error) in enumerate(zip(
        bars, df_sorted['Performance'], df_sorted['Big_Category'], df_sorted['Sub_Error'])):
        
        # 排名标签（左侧）
        ax.text(-0.01, bar.get_y() + bar.get_height()/2, f'#{i+1}',
               ha='right', va='center', fontweight='bold', fontsize=10)
        
        # 性能数值（右侧）
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'{performance:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
    
    # 设置y轴标签
    labels = []
    for category, sub_error in zip(df_sorted['Big_Category'], df_sorted['Sub_Error']):
        if category == sub_error:
            label = category.replace('-Related Errors', '').replace(' Errors', '')
        else:
            cat_short = category.replace('-Related Errors', '').replace(' Errors', '')
            label = f"{cat_short}: {sub_error}"
        labels.append(label)
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f'{metric_key or "Performance"}', fontsize=12, fontweight='bold')
    ax.set_title(f'{METHOD_NAME}: Complete Performance Ranking\n(Red: Bottom 25%, Yellow: Middle 50%, Green: Top 25%)', 
                fontsize=14, fontweight='bold')
    
    # 添加性能区间分割线
    quartiles = [n_items * 0.25, n_items * 0.75]
    for q in quartiles:
        ax.axhline(y=q-0.5, color='black', linestyle=':', alpha=0.5)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_summary_statistics(df, category_performance, metric_key="Performance"):
    """打印详细统计摘要"""
    print(f"\n{'='*60}")
    print(f"  {METHOD_NAME} - 性能分析报告")
    print(f"{'='*60}")
    
    # 基础统计
    print(f"\n📊 基础统计信息:")
    print(f"   总错误类型数量: {len(df)} 个")
    print(f"   大类数量: {len(category_performance)} 个")
    print(f"   整体平均{metric_key or '性能'}: {df['Performance'].mean():.4f}")
    print(f"   性能中位数: {df['Performance'].median():.4f}")
    print(f"   性能标准差: {df['Performance'].std():.4f}")
    print(f"   最高性能: {df['Performance'].max():.4f}")
    print(f"   最低性能: {df['Performance'].min():.4f}")
    
    # 性能分布
    print(f"\n📈 性能分布:")
    q25 = df['Performance'].quantile(0.25)
    q50 = df['Performance'].quantile(0.50)
    q75 = df['Performance'].quantile(0.75)
    print(f"   25%分位数: {q25:.4f}")
    print(f"   50%分位数: {q50:.4f}")
    print(f"   75%分位数: {q75:.4f}")
    
    # 各大类性能排名
    print(f"\n🏆 大类性能排名 (从高到低):")
    for i, (category, performance) in enumerate(category_performance.sort_values(ascending=False).items(), 1):
        category_display = category.replace('-Related Errors', '').replace(' Errors', '')
        print(f"   {i:2d}. {category_display:20s}: {performance:.4f}")
    
    # 识别问题区域
    worst_categories = category_performance.head(3)
    best_categories = category_performance.tail(3)
    
    print(f"\n🔴 需要重点关注的问题区域:")
    for category, performance in worst_categories.items():
        category_display = category.replace('-Related Errors', '').replace(' Errors', '')
        count = len(df[df['Big_Category'] == category])
        print(f"   - {category_display}: {performance:.4f} ({count}个子类型)")
    
    print(f"\n🟢 相对表现较好的区域:")
    for category, performance in best_categories.items():
        category_display = category.replace('-Related Errors', '').replace(' Errors', '')
        count = len(df[df['Big_Category'] == category])
        print(f"   - {category_display}: {performance:.4f} ({count}个子类型)")

def main():
    """主函数"""
    print("Loading baseline data...")
    df, categories, metric_key = load_baseline_data()
    
    print(f"Found {len(df)} error types across {len(categories)} categories")
    print(f"Detected performance metric: {metric_key or 'Performance'}")
    
    print("\n1. Creating performance overview...")
    category_performance = create_performance_overview(df, categories, metric_key)
    
    print("\n2. Creating detailed breakdown...")
    create_detailed_breakdown(df, category_performance)
    
    print("\n3. Creating performance matrix...")
    create_performance_matrix(df)
    
    print("\n4. Creating performance ranking...")
    create_performance_ranking(df, metric_key)
    
    print("\n5. Generating summary statistics...")
    print_summary_statistics(df, category_performance, metric_key)

if __name__ == "__main__":
    main()