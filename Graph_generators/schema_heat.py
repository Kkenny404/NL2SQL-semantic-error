# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # ==== 配置 ====
# ALL_ERRORS_FILE = "Spider/error_explaination.json"
# FILES = {
#     # "Baseline(Schema without Primary and Foreign Keys)": "results/SSS/eval_results/erros/Basline_error.json",
#     # "With Parsed SQL": "results/SSS/eval_results/erros/parsed_sql_error.json",
#     "Schema without Keys Info\n (Baseline)": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
#     "Schema with Keys Info": "results/eval_result/errors/KeyInfo_GPT_results_error.json",
#     "No Schema Provided": "results/eval_result/errors/no_schema_error.json",
# }

# # # ==== 名称映射 ====
# name_mapping = {
#     "ASC_DESC": "ASC/DESC",
#     "DateTime Functions": "Date/Time Functions",
#     "Comparison Operator": "Comparison Operator Mismatch",
#     "Logical Operator": "Logical Operator Mismatch",
#     "Others": "Other"
# }


# # 大类到颜色的映射 - 更明显的颜色
# CATEGORY_COLORS = {
#     'Function-Related Errors': '#D32F2F',      # 深红色
#     'Clause-Related Errors': '#1976D2',       # 深蓝色
#     'Attribute-Related Errors': '#388E3C',    # 深绿色
#     'Condition-Related Errors': '#F57C00',    # 深橙色
#     'Value-Related Errors': '#7B1FA2',        # 深紫色
#     'Table-Related Errors': '#C2185B',        # 深粉色
#     'Operator-Related Errors': '#00796B',     # 深青色
#     'Subquery-Related Errors': '#5D4037',     # 深棕色
#     'Other Errors': '#455A64'                 # 深灰色
# }


# # ==== 1. 读取全集错误类型并建立映射 ====
# with open(ALL_ERRORS_FILE) as f:
#     categories = json.load(f)

# # 从你的数据文件中读取一个样本来获取大类结构
# sample_file = list(FILES.values())[0]
# with open(sample_file) as f:
#     sample_data = json.load(f)

# # 建立子错误到大类的映射
# sub_error_to_category = {}
# ordered_categories = []
# ordered_sub_errors = []

# for big_category, sub_errors_dict in sample_data.items():
#     if big_category == "average_recall":
#         continue
#     ordered_categories.append(big_category)
#     for sub_error in sub_errors_dict.keys():
#         if sub_error != "average_recall":
#             sub_error_to_category[sub_error] = big_category
#             ordered_sub_errors.append(sub_error)

# # ==== 2. 读取方法结果并整合 ====
# data = []
# for method, path in FILES.items():
#     with open(path) as f:
#         res = json.load(f)
    
#     for big_category, sub_errors_dict in res.items():
#         if big_category == "average_recall":
#             continue
#         for sub_error, metrics in sub_errors_dict.items():
#             if sub_error != "average_recall":
#                 recall = metrics.get("recall", None)
#                 data.append({
#                     "Method": method,
#                     "Error Type": sub_error,
#                     "Big Category": big_category,
#                     "Recall": recall
#                 })

# df = pd.DataFrame(data)

# # ==== 3. 图1: 大类概览 ====
# # 计算大类平均性能
# category_data = []
# for method in FILES.keys():
#     method_data = df[df['Method'] == method]
#     for category in ordered_categories:
#         cat_data = method_data[method_data['Big Category'] == category]
#         avg_recall = cat_data['Recall'].mean()
#         category_data.append({
#             'Method': method,
#             'Category': category,
#             'Average Recall': avg_recall
#         })

# cat_df = pd.DataFrame(category_data)
# cat_pivot = cat_df.pivot(index="Category", columns="Method", values="Average Recall")

# # 按平均性能排序
# cat_pivot['mean_perf'] = cat_pivot.mean(axis=1)
# cat_pivot_sorted = cat_pivot.sort_values('mean_perf', ascending=False)
# cat_pivot_for_plot = cat_pivot_sorted.drop('mean_perf', axis=1)

# plt.figure(figsize=(12, 8))
# sns.heatmap(cat_pivot_for_plot, annot=True, cmap="RdYlGn", cbar=True,
#             linewidths=1, fmt=".3f", vmin=0, vmax=1,
#             annot_kws={"size": 12})
# plt.title("Error Category Performance Overview\n(NL2SQL-BUGs Dataset)", 
#           fontsize=14, fontweight='bold', pad=20)
# plt.ylabel("Error Category", fontsize=12, fontweight='bold')
# plt.xlabel("Method", fontsize=12, fontweight='bold')
# plt.xticks(rotation=0, ha='center')
# plt.tight_layout()
# plt.show()

# # ==== 4. 图2: 详细子类热力图（大类标签在顶部图例）====
# pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
# pivot_sub = pivot_sub.reindex(ordered_sub_errors)

# # 按大类分组并在组内按性能排序
# grouped_errors = []
# category_positions = {}  # 记录每个大类的位置信息

# current_pos = 0
# for category in ordered_categories:
#     category_errors = [err for err in ordered_sub_errors if sub_error_to_category.get(err) == category]
#     if category_errors:  # 确保不为空
#         category_subset = pivot_sub.loc[category_errors]
#         category_subset['mean_perf'] = category_subset.mean(axis=1)
#         category_subset_sorted = category_subset.sort_values('mean_perf', ascending=False)
#         grouped_errors.extend(category_subset_sorted.index.tolist())
        
#         # 记录大类的位置信息
#         category_positions[category] = {
#             'start': current_pos,
#             'end': current_pos + len(category_errors),
#             'mid': current_pos + len(category_errors) / 2,
#             'errors': category_subset_sorted.index.tolist()
#         }
#         current_pos += len(category_errors)

# # 重新排序
# pivot_sub_grouped = pivot_sub.reindex(grouped_errors)

# fig, ax = plt.subplots(figsize=(14, 18))

# # 绘制热力图
# sns.heatmap(pivot_sub_grouped, annot=True, cmap="RdYlGn", cbar=True,
#             linewidths=0.5, fmt=".2f", vmin=0, vmax=1,
#             annot_kws={"size": 9}, cbar_kws={"shrink": 0.8}, ax=ax)

# plt.title("Detailed Error Type Recall Analysis\n(NL2SQL-BUGs Dataset)", 
#           fontsize=14, fontweight='bold', pad=40)  # 增加pad给图例留空间
# plt.ylabel("Sub Error Type", fontsize=12, fontweight='bold')
# plt.xlabel("Method", fontsize=12, fontweight='bold')
# plt.xticks(rotation=0, ha='cen')

# # 添加大类分割线
# for category, pos_info in category_positions.items():
#     if pos_info['start'] > 0:
#         ax.axhline(y=pos_info['start'], color='black', linewidth=2)

# # 设置y轴标签颜色
# y_labels = []
# y_colors = []
# for error in grouped_errors:
#     category = sub_error_to_category.get(error, "Unknown")
#     y_labels.append(error)
#     y_colors.append(CATEGORY_COLORS.get(category, 'black'))

# ax.set_yticklabels(y_labels)
# for i, (label, color) in enumerate(zip(ax.get_yticklabels(), y_colors)):
#     label.set_color(color)
#     label.set_fontweight('bold')

# # 在顶部添加图例说明大类颜色
# legend_elements = []
# for category in ordered_categories:
#     category_short = category.replace(' Errors', '').replace('-Related', '')
#     legend_elements.append(plt.Rectangle((0,0),1,1, 
#                                        facecolor='white',
#                                        edgecolor=CATEGORY_COLORS.get(category, 'black'),
#                                        linewidth=2,
#                                        label=category_short))

# plt.legend(handles=legend_elements, 
#           loc='upper center', 
#           bbox_to_anchor=(0.5, 1.05),
#           ncol=5, 
#           frameon=True,
#           fontsize=10,
#           title="Error Categories",
#           title_fontsize=11)

# plt.tight_layout()
# plt.show()

# # ==== 5. 打印统计分析 ====
# print("\n=== 大类性能分析 (按平均性能排序) ===")
# for category in cat_pivot_sorted.index:
#     baseline_perf = cat_pivot_sorted.loc[category, 'mean_perf']
#     print(f"{category}: {baseline_perf:.3f}")

# print(f"\n=== 最难检测的错误类型 ===")
# pivot_sub_grouped['mean_perf'] = pivot_sub_grouped.mean(axis=1)
# worst_errors = pivot_sub_grouped.nsmallest(5, 'mean_perf')
# for error in worst_errors.index:
#     category = sub_error_to_category.get(error, "Unknown")
#     mean_perf = worst_errors.loc[error, 'mean_perf']
#     print(f"{error} ({category}): {mean_perf:.3f}")

# print(f"\n=== 最稳定的错误类型 ===")
# pivot_sub_grouped['std_perf'] = pivot_sub_grouped.std(axis=1)
# stable_errors = pivot_sub_grouped.nsmallest(5, 'std_perf')
# for error in stable_errors.index:
#     category = sub_error_to_category.get(error, "Unknown")
#     std_perf = stable_errors.loc[error, 'std_perf']
#     mean_perf = stable_errors.loc[error, 'mean_perf']
#     print(f"{error} ({category}): {mean_perf:.3f} (±{std_perf:.3f})")
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ==== 动态配置 ====
ALL_ERRORS_FILE = "Spider/error_explaination.json"
NL2SQL_BUGS_FILES = {
    # "No Schema": "results/eval_result/errors/no_schema_error.json",
    # "With Keys": "results/eval_result/errors/KeyInfo_GPT_results_error.json", 
    # "Baseline": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
    # "Original SQL Only (Baseline)": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
    # "With Extra Parsed SQL": "results/eval_result/errors/parserBUGS_GPT4.1_error_20250804_140520.json",
    "Simple (Baseline)": "results/eval_result/errors/Baseline_BUGS_GPT4.1_results_1_error.json",
    # "Chain-of-Thought": "results/eval_result/errors/CoT_error.json",
    # "Self-Reflection": "results/eval_result/errors/selfF_error.json",
}

# 名称映射 - 用于显示更友好的名称
METHOD_NAME_MAPPING = {
    "No Schema": "No Schema Provided",
    "With Keys": "Schema with Keys",
    "Baseline": "Baseline Configuration",
    "Original SQL Only (Baseline)": "Original SQL Only (Baseline)",
    "With Extra Parsed SQL": "With Extra Parsed SQL"
}

# 错误类别名称映射 - 统一格式
CATEGORY_NAME_MAPPING = {
    'Function-Related Errors': 'Function-related Errors',
    'Clause-Related Errors': 'Clause-related Errors', 
    'Attribute-Related Errors': 'Attribute-related Errors',
    'Condition-Related Errors': 'Condition-related Errors',
    'Value-Related Errors': 'Value-related Errors',
    'Table-Related Errors': 'Table-related Errors',
    'Operator-Related Errors': 'Operator-related Errors',
    'Subquery-Related Errors': 'Subquery-related Errors',
    'Other Errors': 'Other Errors'
}

# NL2SQL-BUGs数据集专用配色方案
CATEGORY_COLORS = {
    'Function-related Errors': '#E53E3E',      # 现代红色
    'Clause-related Errors': '#3182CE',       # 现代蓝色
    'Attribute-related Errors': '#38A169',    # 现代绿色
    'Condition-related Errors': '#DD6B20',    # 现代橙色
    'Value-related Errors': '#805AD5',        # 现代紫色
    'Table-related Errors': '#D53F8C',        # 现代粉色
    'Operator-related Errors': '#319795',     # 现代青色
    'Subquery-related Errors': '#A0AEC0',     # 现代灰蓝色
    'Other Errors': '#4A5568'                 # 现代深灰色
}

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

# 指定方法的显示顺序
DESIRED_METHOD_ORDER = ["Simple (Baseline)", "Chain-of-Thought", "Self-Reflection"]
# ==== 动态数据读取和预处理 ====
def load_and_process_nl2sql_bugs_data():
    """动态加载和处理NL2SQL-BUGs数据，自动提取所有可用的方法和错误类型"""
    
    # 首先从一个示例文件中获取错误类型结构
    sample_file = list(NL2SQL_BUGS_FILES.values())[0]
    with open(sample_file) as f:
        sample_data = json.load(f)
    
    # 自动提取大类和子错误类型，并应用名称映射
    sub_error_to_category = {}
    ordered_categories = []
    ordered_sub_errors = []
    
    for big_category, sub_errors_dict in sample_data.items():
        if big_category == "average_recall":  # 跳过总体统计
            continue
        
        # 应用类别名称映射
        mapped_category = CATEGORY_NAME_MAPPING.get(big_category, big_category)
        ordered_categories.append(mapped_category)
        
        for sub_error in sub_errors_dict.keys():
            if sub_error != "average_recall":
                sub_error_to_category[sub_error] = mapped_category
                ordered_sub_errors.append(sub_error)
    
    # 使用指定的方法顺序
    available_methods = DESIRED_METHOD_ORDER
    
    # 读取所有方法的数据
    all_methods_data = {}
    data = []
    
    for method in DESIRED_METHOD_ORDER:
        if method not in NL2SQL_BUGS_FILES:
            print(f"Warning: Method '{method}' not found in files")
            continue
            
        path = NL2SQL_BUGS_FILES[method]
        mapped_method = METHOD_NAME_MAPPING.get(method, method)
        
        with open(path) as f:
            method_data = json.load(f)
        all_methods_data[mapped_method] = method_data
        
        for big_category, sub_errors_dict in method_data.items():
            if big_category == "average_recall":
                continue
            
            # 应用类别名称映射
            mapped_category = CATEGORY_NAME_MAPPING.get(big_category, big_category)
            
            for sub_error, metrics in sub_errors_dict.items():
                if sub_error != "average_recall":
                    recall = metrics.get("recall", None)
                    data.append({
                        "Method": mapped_method,
                        "Error Type": sub_error,
                        "Big Category": mapped_category,
                        "Recall": recall
                    })
    
    df = pd.DataFrame(data)
    
    # 确保方法按指定顺序显示
    ordered_methods = [METHOD_NAME_MAPPING.get(method, method) for method in DESIRED_METHOD_ORDER if method in NL2SQL_BUGS_FILES]
    
    # 返回处理后的数据和元信息
    return df, sub_error_to_category, ordered_categories, ordered_sub_errors, ordered_methods, all_methods_data


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
df, sub_error_to_category, ordered_categories, ordered_sub_errors, available_methods, all_methods_data = load_and_process_nl2sql_bugs_data()

# 动态生成颜色方案
DYNAMIC_CATEGORY_COLORS = generate_dynamic_colors(ordered_categories)

print(f"Detected Methods: {available_methods}")
print(f"Detected Categories: {ordered_categories}")
print(f"Total Sub-errors: {len(ordered_sub_errors)}")

# ==== 图1: NL2SQL-BUGs大类概览热力图 ====
def create_nl2sql_bugs_category_overview():
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
    
    # 按平均性能排序
    cat_pivot['mean_perf'] = cat_pivot.mean(axis=1)
    cat_pivot_sorted = cat_pivot.sort_values('mean_perf', ascending=True)
    cat_pivot_for_plot = cat_pivot_sorted.drop('mean_perf', axis=1)
    
    # 动态调整图形大小
    fig_width = max(10, len(available_methods) * 2)
    fig_height = max(8, len(ordered_categories) * 0.8)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 使用动态范围
    vmin = max(0.6, cat_pivot_for_plot.min().min() * 0.9)
    vmax = min(1.0, cat_pivot_for_plot.max().max() * 1.05)
    
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
    
    ax.set_title("NL2SQL-BUGs Dataset: Baseline Error Category Performance Overview", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Error Category", fontsize=13, fontweight='bold')
    ax.set_xlabel("Method", fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    
    # 动态添加性能变化指示器（仅当有多个方法时）
    if len(available_methods) >= 2:
        # 寻找包含"Baseline"或"Schema with Keys"的方法进行对比
        baseline_method = None
        comparison_method = None
        
        for method in available_methods:
            if "Baseline" in method:
                baseline_method = method
            elif "Schema with Keys" in method or "With Keys" in method:
                comparison_method = method
        
        if baseline_method and comparison_method:
            for i, category in enumerate(cat_pivot_for_plot.index):
                baseline_val = cat_pivot_for_plot.loc[category, baseline_method]
                comparison_val = cat_pivot_for_plot.loc[category, comparison_method]
                change = comparison_val - baseline_val
                
                if abs(change) > 0.02:  # 显著变化
                    col_idx = list(cat_pivot_for_plot.columns).index(comparison_method)
                    if change > 0:
                        ax.annotate('↗', xy=(col_idx + 0.5, i + 0.5), xytext=(col_idx + 0.3, i + 0.3),
                                   fontsize=18, color='green', weight='bold',
                                   ha='center', va='center')
                    else:
                        ax.annotate('↘', xy=(col_idx + 0.5, i + 0.5), xytext=(col_idx + 0.3, i + 0.7),
                                   fontsize=18, color='red', weight='bold',
                                   ha='center', va='center')
    
    plt.tight_layout()
    return fig

# ==== 图2: NL2SQL-BUGs详细热力图 ====
def create_nl2sql_bugs_detailed_heatmap():
    pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
    pivot_sub = pivot_sub.reindex(ordered_sub_errors)
    
    # 确保列按指定顺序排列
    pivot_sub = pivot_sub.reindex(columns=available_methods)
    
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
    # 确保列仍然按指定顺序排列
    pivot_sub_grouped = pivot_sub_grouped.reindex(columns=available_methods)
    
    # 动态调整图形大小
    fig_width = max(8, len(available_methods) * 3)
    fig_height = max(18, len(grouped_errors) * 0.4)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # 动态确定颜色范围
    vmin = max(0.4, pivot_sub_grouped.min().min() * 0.9)
    vmax = min(1.0, pivot_sub_grouped.max().max() * 1.05)
    
    im = sns.heatmap(pivot_sub_grouped, 
                     annot=True, 
                     cmap="RdYlGn", 
                     cbar=True,
                     linewidths=1, 
                     fmt=".2f", 
                     vmin=vmin, 
                     vmax=vmax,
                     annot_kws={"size": 11},
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
    
    ax.set_title("Baseline -- Detailed Error Type Recall Analysis\n (NL2SQL-BUGs Dataset)", 
                fontsize=16, fontweight='bold', pad=25)
    ax.set_ylabel("Sub Error Type", fontsize=13, fontweight='bold')
    ax.set_xlabel("Method", fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)

    # 添加类别标签
    ax.text(-0.6, 0.5, "Error categories",
            ha='center', va='center', rotation='vertical',
            fontsize=13, fontweight='bold',
            transform=ax.transAxes)
            
    for category, pos_info in category_positions.items():
        y_center = pos_info['start'] + (pos_info['end'] - pos_info['start']) / 2 + 0.5
        category_short = category.replace('-related', '').replace(' Errors', '')
        
        ax.text(-0.5, y_center, category_short,
                rotation=0, ha='center', va='center',
                fontsize=11, fontweight='bold',
                color=DYNAMIC_CATEGORY_COLORS.get(category, 'black'),
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor='white', 
                         edgecolor=DYNAMIC_CATEGORY_COLORS.get(category, 'black'),
                         linewidth=1.5, alpha=0.9))

    plt.subplots_adjust(left=0.2)
    return fig

# ==== 图3: NL2SQL-BUGs性能对比分析 ====
def create_nl2sql_bugs_comparison_analysis():
    """创建动态的性能对比分析，支持任意数量的方法"""
    
    pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
    
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
    ax1.set_title(f'NL2SQL-BUGs: {method} Performance Distribution', fontweight='bold')
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
    ax2.set_title(f'NL2SQL-BUGs: {method} Performance by Category', fontweight='bold')
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
    ax3.set_title(f'NL2SQL-BUGs: {method} Top 10 Performance', fontweight='bold')
    ax3.set_xlabel('Recall Score')
    
    ax4.barh(range(len(bottom_errors)), bottom_errors[method], color='red', alpha=0.7)
    ax4.set_yticks(range(len(bottom_errors)))
    ax4.set_yticklabels([err[:30] + '...' if len(err) > 30 else err for err in bottom_errors.index], fontsize=8)
    ax4.set_title(f'NL2SQL-BUGs: {method} Bottom 10 Performance', fontweight='bold')
    ax4.set_xlabel('Recall Score')
    
    plt.tight_layout()
    return fig

def create_multi_method_comparison(pivot_sub):
    """多方法对比分析"""
    
    # 智能选择对比方法 - 优先选择Baseline和Schema with Keys
    baseline_method = None
    comparison_method = None
    
    for method in available_methods:
        if "Baseline" in method:
            baseline_method = method
        elif "Schema with Keys" in method:
            comparison_method = method
    
    # 如果没有找到预期的方法，使用前两个方法
    if not baseline_method or not comparison_method:
        baseline_method = available_methods[0]
        comparison_method = available_methods[1] if len(available_methods) > 1 else available_methods[0]
    
    differences = {}
    for error_type in pivot_sub.index:
        baseline_val = pivot_sub.loc[error_type, baseline_method]
        comparison_val = pivot_sub.loc[error_type, comparison_method]
        
        differences[error_type] = {
            f'{comparison_method} vs {baseline_method}': comparison_val - baseline_val,
            'Category': sub_error_to_category.get(error_type, 'Other'),
            baseline_method: baseline_val,
            comparison_method: comparison_val
        }
    
    # 转换为DataFrame
    diff_data = []
    for error_type, diffs in differences.items():
        diff_data.append({
            'Error Type': error_type,
            'Difference': diffs[f'{comparison_method} vs {baseline_method}'],
            'Category': diffs['Category'],
            baseline_method: diffs[baseline_method],
            comparison_method: diffs[comparison_method]
        })
    
    diff_df = pd.DataFrame(diff_data)
    
    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 性能差异分布
    for category in ordered_categories:
        cat_data = diff_df[diff_df['Category'] == category]
        if not cat_data.empty:
            ax1.scatter(cat_data[baseline_method], cat_data['Difference'],
                       color=DYNAMIC_CATEGORY_COLORS.get(category, 'gray'),
                       alpha=0.7, s=80, label=category.replace(' Errors', ''))
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel(f'{baseline_method} Performance')
    ax1.set_ylabel(f'Performance Change ({comparison_method} vs {baseline_method})')
    ax1.set_title(f'NL2SQL-BUGs: Performance Change vs {baseline_method} Performance', fontweight='bold')
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
    ax2.set_title(f'NL2SQL-BUGs: Average Change by Category\n({comparison_method} vs {baseline_method})', fontweight='bold')
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
    ax3.set_title('NL2SQL-BUGs: Performance Change Distribution', fontweight='bold')
    
    # 子图4: 统计摘要
    top_improvements = diff_df.nlargest(5, 'Difference')
    top_degradations = diff_df.nsmallest(5, 'Difference')
    
    ax4.axis('off')
    summary_text = f"""NL2SQL-BUGs DATASET PERFORMANCE SUMMARY

Overall Performance:
• {baseline_method}: {diff_df[baseline_method].mean():.3f}
• {comparison_method}: {diff_df[comparison_method].mean():.3f}
• Overall Change: {diff_df['Difference'].mean():.3f}

Top 5 Improvements ({comparison_method} vs {baseline_method}):
"""
    
    for _, row in top_improvements.iterrows():
        summary_text += f"  • {row['Error Type']}: +{row['Difference']:.3f}\n"
    
    summary_text += f"\nTop 5 Degradations:\n"
    for _, row in top_degradations.iterrows():
        summary_text += f"  • {row['Error Type']}: {row['Difference']:.3f}\n"
    
    summary_text += f"""
Statistics:
• Improved: {improved} error types
• Degraded: {degraded} error types  
• Unchanged: {unchanged} error types
• Largest improvement: +{diff_df['Difference'].max():.3f}
• Largest degradation: {diff_df['Difference'].min():.3f}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    return fig

# ==== 图4: NL2SQL-BUGs Top/Bottom 错误类型分析 ====
def create_nl2sql_bugs_top_bottom_analysis():
    pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
    pivot_sub['Mean'] = pivot_sub.mean(axis=1)
    pivot_sub['Std'] = pivot_sub.std(axis=1)
    
    # 如果有多个方法，计算最大改进
    if len(available_methods) >= 2:
        # 智能选择baseline和comparison方法
        baseline_method = None
        comparison_method = None
        
        for method in available_methods:
            if "Baseline" in method:
                baseline_method = method
            elif "Schema with Keys" in method:
                comparison_method = method
        
        if baseline_method and comparison_method:
            pivot_sub['Difference'] = pivot_sub[comparison_method] - pivot_sub[baseline_method]
            most_improved = pivot_sub.nlargest(6, 'Difference')
            most_degraded = pivot_sub.nsmallest(6, 'Difference')
        else:
            most_improved = pivot_sub.nlargest(6, 'Mean')
            most_degraded = pivot_sub.nsmallest(6, 'Mean')
    else:
        most_improved = pivot_sub.nlargest(6, 'Mean')
        most_degraded = pivot_sub.nsmallest(6, 'Mean')
    
    # 获取最难和最易检测的错误
    hardest_errors = pivot_sub.nsmallest(6, 'Mean')
    easiest_errors = pivot_sub.nlargest(6, 'Mean')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # 获取要显示的方法列表（最多3个）
    display_methods = available_methods[:3]
    
    # 最难检测的错误
    hardest_data = hardest_errors[display_methods].T
    im1 = ax1.imshow(hardest_data.values, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(hardest_data.columns)))
    ax1.set_xticklabels(hardest_data.columns, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(hardest_data.index)))
    ax1.set_yticklabels(hardest_data.index)
    ax1.set_title('NL2SQL-BUGs: Hardest to Detect Errors', fontweight='bold')
    
    for i in range(len(hardest_data.index)):
        for j in range(len(hardest_data.columns)):
            ax1.text(j, i, f'{hardest_data.iloc[i, j]:.2f}',
                    ha='center', va='center', fontweight='bold', color='white')
    
    # 最易检测的错误
    easiest_data = easiest_errors[display_methods].T
    im2 = ax2.imshow(easiest_data.values, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(easiest_data.columns)))
    ax2.set_xticklabels(easiest_data.columns, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(easiest_data.index)))
    ax2.set_yticklabels(easiest_data.index)
    ax2.set_title('NL2SQL-BUGs: Easiest to Detect Errors', fontweight='bold')
    
    for i in range(len(easiest_data.index)):
        for j in range(len(easiest_data.columns)):
            ax2.text(j, i, f'{easiest_data.iloc[i, j]:.2f}',
                    ha='center', va='center', fontweight='bold')
    
    # 最大改进/最高性能
    improved_data = most_improved[display_methods].T
    im3 = ax3.imshow(improved_data.values, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(improved_data.columns)))
    ax3.set_xticklabels(improved_data.columns, rotation=45, ha='right', fontsize=9)
    ax3.set_yticks(range(len(improved_data.index)))
    ax3.set_yticklabels(improved_data.index)
    
    if len(available_methods) >= 2 and 'Difference' in pivot_sub.columns:
        ax3.set_title('NL2SQL-BUGs: Most Improved Errors', fontweight='bold')
    else:
        ax3.set_title('NL2SQL-BUGs: Best Performance Errors', fontweight='bold')
    
    for i in range(len(improved_data.index)):
        for j in range(len(improved_data.columns)):
            ax3.text(j, i, f'{improved_data.iloc[i, j]:.2f}',
                    ha='center', va='center', fontweight='bold')
    
    # 最大退化/最低性能
    degraded_data = most_degraded[display_methods].T
    im4 = ax4.imshow(degraded_data.values, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(degraded_data.columns)))
    ax4.set_xticklabels(degraded_data.columns, rotation=45, ha='right', fontsize=9)
    ax4.set_yticks(range(len(degraded_data.index)))
    ax4.set_yticklabels(degraded_data.index)
    
    if len(available_methods) >= 2 and 'Difference' in pivot_sub.columns:
        ax4.set_title('NL2SQL-BUGs: Most Degraded Errors', fontweight='bold')
    else:
        ax4.set_title('NL2SQL-BUGs: Lowest Performance Errors', fontweight='bold')
    
    for i in range(len(degraded_data.index)):
        for j in range(len(degraded_data.columns)):
            ax4.text(j, i, f'{degraded_data.iloc[i, j]:.2f}',
                    ha='center', va='center', fontweight='bold')
    
    plt.suptitle('NL2SQL-BUGs Dataset: Top/Bottom Error Type Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# ==== 生成所有NL2SQL-BUGs图表 ====
if __name__ == "__main__":
    print("Generating NL2SQL-BUGs Dataset visualizations...")
    print(f"Processing {len(available_methods)} methods: {available_methods}")
    print(f"Processing {len(ordered_categories)} categories: {ordered_categories}")
    print(f"Processing {len(ordered_sub_errors)} sub-error types")
    
    # 创建图表
    fig1 = create_nl2sql_bugs_category_overview()
    fig1.savefig('nl2sql_bugs_category_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig2 = create_nl2sql_bugs_detailed_heatmap()
    fig2.savefig('nl2sql_bugs_detailed_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig3 = create_nl2sql_bugs_comparison_analysis()
    fig3.savefig('nl2sql_bugs_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig4 = create_nl2sql_bugs_top_bottom_analysis()
    fig4.savefig('nl2sql_bugs_top_bottom_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All NL2SQL-BUGs Dataset visualizations have been generated and saved!")
    
    # 打印统计信息
    print("\n=== NL2SQL-BUGs Dataset Statistics ===")
    pivot_sub = df.pivot(index="Error Type", columns="Method", values="Recall")
    
    for method in available_methods:
        avg_performance = pivot_sub[method].mean()
        print(f"{method} Average: {avg_performance:.3f}")
    
    if len(available_methods) >= 2:
        # 找到baseline和comparison方法
        baseline_method = None
        comparison_method = None
        
        for method in available_methods:
            if "Baseline" in method:
                baseline_method = method
            elif "Schema with Keys" in method:
                comparison_method = method
        
        if baseline_method and comparison_method:
            avg_diff = (pivot_sub[comparison_method] - pivot_sub[baseline_method]).mean()
            improved_count = (pivot_sub[comparison_method] > pivot_sub[baseline_method]).sum()
            degraded_count = (pivot_sub[comparison_method] < pivot_sub[baseline_method]).sum()
            
            print(f"{comparison_method} vs {baseline_method} Average Change: {avg_diff:+.3f}")
            print(f"Improved Error Types: {improved_count}")
            print(f"Degraded Error Types: {degraded_count}")
        
        # 计算所有方法间的平均差异
        for i in range(len(available_methods)):
            for j in range(i+1, len(available_methods)):
                method1, method2 = available_methods[i], available_methods[j]
                avg_diff = (pivot_sub[method2] - pivot_sub[method1]).mean()
                print(f"{method2} vs {method1} Average Change: {avg_diff:+.3f}")