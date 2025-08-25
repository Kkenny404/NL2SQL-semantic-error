# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle
# import matplotlib.patches as mpatches

# # ==== SSS数据集配置 ====
# ALL_ERRORS_FILE = "Spider/error_explaination.json"  # 或者SSS对应的错误文件
# SSS_FILES = {
#     "Baseline": "results/SSS/eval_results/erros/Basline_error.json",  # 修改为实际SSS baseline文件路径
#     "With Extra Parsed SQL": "results/SSS/eval_results/erros/parsed_sql_error.json",  # 修改为实际SSS no schema文件路径
# }

# # ==== SSS数据集不需要名称映射，直接使用原始名称 ====
# # name_mapping = {}  # SSS数据集不使用映射

# # 学术风格颜色映射
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

# # SSS对比方法的颜色
# SSS_COMPARISON_COLORS = {
#     'With Extra Parsed SQL': {'color': '#FF6B35', 'label': 'With Extra Parsed SQL'},
# }

# def load_and_process_sss_data():
#     """加载和处理SSS错误数据"""
#     # 读取全集错误类型
#     with open(ALL_ERRORS_FILE) as f:
#         categories = json.load(f)
    
#     # 从baseline文件获取大类结构
#     baseline_file = SSS_FILES["Baseline"]
#     with open(baseline_file) as f:
#         sample_data = json.load(f)
    
#     # 建立子错误到大类的映射
#     sub_error_to_category = {}
#     ordered_categories = []
#     ordered_sub_errors = []
    
#     for big_category, sub_errors_dict in sample_data.items():
#         if big_category == "average_recall":
#             continue
#         ordered_categories.append(big_category)
#         for sub_error in sub_errors_dict.keys():
#             if sub_error != "average_recall":
#                 sub_error_to_category[sub_error] = big_category
#                 ordered_sub_errors.append(sub_error)
    
#     # 读取方法结果并整合
#     data = []
#     for method, path in SSS_FILES.items():
#         with open(path) as f:
#             res = json.load(f)
        
#         for big_category, sub_errors_dict in res.items():
#             if big_category == "average_recall":
#                 continue
#             for sub_error, metrics in sub_errors_dict.items():
#                 if sub_error != "average_recall":
#                     recall = metrics.get("recall", None)
#                     data.append({
#                         "Method": method,
#                         "Error Type": sub_error,
#                         "Big Category": big_category,
#                         "Recall": recall
#                     })
    
#     return pd.DataFrame(data), sub_error_to_category, ordered_categories, ordered_sub_errors

# def create_sss_baseline_comparison_chart(df, ordered_categories):
#     """创建SSS数据集的Baseline对比图"""
#     baseline_method = "Baseline"
#     comparison_methods = ["With Extra Parsed SQL"]
    
#     # 计算大类平均性能
#     category_data = []
#     for method in SSS_FILES.keys():
#         method_data = df[df['Method'] == method]
#         for category in ordered_categories:
#             cat_data = method_data[method_data['Big Category'] == category]
#             avg_recall = cat_data['Recall'].mean()
#             category_data.append({
#                 'Method': method,
#                 'Category': category,
#                 'Average Recall': avg_recall
#             })
    
#     cat_df = pd.DataFrame(category_data)
    
#     # 获取baseline性能
#     baseline_df = cat_df[cat_df['Method'] == baseline_method].set_index('Category')['Average Recall']
    
#     # 计算相对于baseline的差异
#     comparison_data = []
#     for method in comparison_methods:
#         method_df = cat_df[cat_df['Method'] == method].set_index('Category')['Average Recall']
#         for category in ordered_categories:
#             if category in baseline_df.index and category in method_df.index:
#                 baseline_val = baseline_df[category]
#                 method_val = method_df[category]
#                 if baseline_val > 0:  # 避免除零
#                     difference = method_val - baseline_val
#                     percentage_diff = (difference / baseline_val) * 100
#                     comparison_data.append({
#                         'Method': method,
#                         'Category': category,
#                         'Difference': difference,
#                         'Percentage_Diff': percentage_diff,
#                         'Baseline_Value': baseline_val,
#                         'Method_Value': method_val
#                     })
    
#     comp_df = pd.DataFrame(comparison_data)
    
#     # 按baseline性能排序大类（从低到高）
#     baseline_sorted = baseline_df.sort_values(ascending=True)
#     ordered_cats = baseline_sorted.index.tolist()
    
#     # 简化大类名称
#     category_display_names = {cat: cat.replace('-Related Errors', '').replace(' Errors', '') 
#                             for cat in ordered_cats}
    
#     # 创建图形
#     fig, ax = plt.subplots(figsize=(12, 10))
    
#     # 设置y轴位置 - 只有一个对比方法，所以条形图更简单
#     y_pos = np.arange(len(ordered_cats))
#     height = 0.6  # 增加高度因为只有一个方法
    
#     # 绘制对比条形图
#     method = comparison_methods[0]
#     method_data = comp_df[comp_df['Method'] == method]
    
#     differences = []
#     baseline_values = []
#     method_values = []
    
#     for cat in ordered_cats:
#         cat_data = method_data[method_data['Category'] == cat]
#         if not cat_data.empty:
#             diff = cat_data['Difference'].iloc[0]
#             differences.append(diff)
#             baseline_values.append(cat_data['Baseline_Value'].iloc[0])
#             method_values.append(cat_data['Method_Value'].iloc[0])
#         else:
#             differences.append(0)
#             baseline_values.append(0)
#             method_values.append(0)
    
#     # 绘制水平条形图
#     bars = ax.barh(y_pos, differences, height,
#                   label=SSS_COMPARISON_COLORS[method]['label'],
#                   color=SSS_COMPARISON_COLORS[method]['color'],
#                   alpha=0.8,
#                   edgecolor='black',
#                   linewidth=0.8)
    
#     # 添加数值标签
#     for j, (bar, diff, baseline_val, method_val) in enumerate(zip(bars, differences, baseline_values, method_values)):
#         if abs(diff) > 0.001:  # 只显示有意义的差异
#             # 计算标签位置
#             label_x = bar.get_width() + (0.01 if diff >= 0 else -0.01)
#             ha = 'left' if diff >= 0 else 'right'
            
#             # 创建标签文本
#             pct_change = (diff / baseline_val * 100) if baseline_val > 0 else 0
#             label_text = f'{diff:+.3f}\n({pct_change:+.1f}%)'
            
#             ax.text(label_x, bar.get_y() + bar.get_height()/2,
#                    label_text, ha=ha, va='center', fontsize=10, fontweight='bold')
    
#     # 添加baseline参考线（x=0）
#     ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Baseline Reference')
    
#     # 设置标签和标题
#     ax.set_xlabel('Difference from Baseline (Negative Recall Score)', fontsize=14, fontweight='bold')
#     ax.set_ylabel('Error Categories', fontsize=14, fontweight='bold')
#     ax.set_title('Input Adjustment on SQL Info -- Performance Comparison Relative to Baseline \n (SSS Dataset)', 
#                 fontsize=16, fontweight='bold', pad=20)
    
#     # 设置y轴标签并着色
#     ax.set_yticks(y_pos)
#     y_labels = [category_display_names[cat] for cat in ordered_cats]
#     ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    
#     # 为y轴标签着色
#     for i, (label, cat) in enumerate(zip(ax.get_yticklabels(), ordered_cats)):
#         label.set_color(CATEGORY_COLORS.get(cat, 'black'))
    
#     # 设置x轴
#     max_abs_diff = max([abs(d) for d in differences] + [0.1])
#     ax.set_xlim(-max_abs_diff * 1.4, max_abs_diff * 1.4)
#     ax.grid(axis='x', alpha=0.3, linestyle='--')
    
#     # 添加图例
#     legend_elements = [
#         plt.Rectangle((0,0),1,1, facecolor=SSS_COMPARISON_COLORS['With Extra Parsed SQL']['color'], 
#                      alpha=0.8, label='With Extra Parsed SQL'),
#         plt.Line2D([0], [0], color='black', linewidth=2, label='Original SQL Only\n (Baseline)'),
#     ]
    
#     ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)
    
#     # 添加baseline值信息
#     baseline_text = "Baseline Performance (SSS):\n"
#     for cat in ordered_cats:
#         baseline_val = baseline_sorted[cat]
#         cat_display = category_display_names[cat]
#         baseline_text += f"{cat_display}: {baseline_val:.3f}\n"
    
#     ax.text(0.02, 0.98, baseline_text, transform=ax.transAxes, fontsize=10,
#            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
#     plt.tight_layout()
#     plt.show()
    
#     return comp_df, ordered_cats, baseline_sorted

# def create_sss_detailed_comparison(df, sub_error_to_category, ordered_cats, baseline_sorted):
#     """创建SSS数据集的详细子错误类型对比图"""
#     baseline_method = "Baseline"
#     comparison_methods = ["With Extra Parsed SQL"]
    
#     # 选择表现最差的3个大类进行详细分析
#     worst_categories = list(baseline_sorted.head(3).index)
    
#     fig, axes = plt.subplots(len(worst_categories), 1, figsize=(14, 6*len(worst_categories)))
#     if len(worst_categories) == 1:
#         axes = [axes]
    
#     for idx, category in enumerate(worst_categories):
#         ax = axes[idx]
        
#         # 获取该大类下的所有子错误
#         sub_errors = [err for err, cat in sub_error_to_category.items() if cat == category]
#         category_data = df[df['Big Category'] == category]
        
#         # 获取baseline性能
#         baseline_data = category_data[category_data['Method'] == baseline_method]
#         baseline_sub_performance = baseline_data.set_index('Error Type')['Recall']
        
#         # 计算对比数据
#         comparison_sub_data = []
#         for method in comparison_methods:
#             method_data = category_data[category_data['Method'] == method]
#             method_sub_performance = method_data.set_index('Error Type')['Recall']
            
#             for sub_error in sub_errors:
#                 if sub_error in baseline_sub_performance.index and sub_error in method_sub_performance.index:
#                     baseline_val = baseline_sub_performance[sub_error]
#                     method_val = method_sub_performance[sub_error]
#                     difference = method_val - baseline_val
                    
#                     comparison_sub_data.append({
#                         'Method': method,
#                         'Sub_Error': sub_error,
#                         'Difference': difference,
#                         'Baseline_Value': baseline_val,
#                         'Method_Value': method_val
#                     })
        
#         comp_sub_df = pd.DataFrame(comparison_sub_data)
        
#         # 按baseline性能排序子错误
#         sorted_sub_errors = baseline_sub_performance.sort_values(ascending=True).index.tolist()
#         sorted_sub_errors = [err for err in sorted_sub_errors if err in sub_errors]
        
#         if not sorted_sub_errors:
#             continue
        
#         # 设置y轴位置
#         y_pos = np.arange(len(sorted_sub_errors))
#         height = 0.6
        
#         # 绘制对比条形图
#         method = comparison_methods[0]
#         method_data = comp_sub_df[comp_sub_df['Method'] == method]
        
#         differences = []
        
#         for sub_error in sorted_sub_errors:
#             sub_data = method_data[method_data['Sub_Error'] == sub_error]
#             if not sub_data.empty:
#                 diff = sub_data['Difference'].iloc[0]
#                 differences.append(diff)
#             else:
#                 differences.append(0)
        
#         # 绘制条形图
#         bars = ax.barh(y_pos, differences, height,
#                       color=SSS_COMPARISON_COLORS[method]['color'],
#                       alpha=0.8,
#                       edgecolor='black',
#                       linewidth=0.8,
#                       label=SSS_COMPARISON_COLORS[method]['label'])
        
#         # 添加数值标签
#         for j, (bar, diff, sub_error) in enumerate(zip(bars, differences, sorted_sub_errors)):
#             if abs(diff) > 0.001:
#                 label_x = bar.get_width() + (0.005 if diff >= 0 else -0.005)
#                 ha = 'left' if diff >= 0 else 'right'
#                 ax.text(label_x, bar.get_y() + bar.get_height()/2,
#                        f'{diff:+.3f}', ha=ha, va='center', fontsize=9, fontweight='bold')
        
#         # 添加baseline参考线
#         ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        
#         # 设置标签和标题
#         category_display = category.replace('-Related Errors', '').replace(' Errors', '')
#         ax.set_title(f'SSS - {category_display}: Detailed Comparison vs Baseline', 
#                     fontsize=14, fontweight='bold', color=CATEGORY_COLORS.get(category, 'black'))
        
#         if idx == len(worst_categories) - 1:
#             ax.set_xlabel('Difference from Baseline (Negative Recall Score)', fontsize=12, fontweight='bold')
        
#         # 直接使用原始的子错误名称（SSS不需要映射）
#         display_sub_errors = sorted_sub_errors
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(display_sub_errors, fontsize=10)
        
#         # 设置x轴范围
#         if len(differences) > 0:
#             max_abs_diff = max([abs(d) for d in differences if d != 0] + [0.05])
#             ax.set_xlim(-max_abs_diff * 1.3, max_abs_diff * 1.3)
        
#         ax.grid(axis='x', alpha=0.3, linestyle='--')
        
#         if idx == 0:  # 只在第一个子图显示图例
#             ax.legend(fontsize=11, loc='upper right')
    
#     plt.suptitle('SSS Dataset - Detailed Error Type Comparison vs Baseline\n(Worst Performing Categories)', 
#                 fontsize=16, fontweight='bold', y=0.98)
#     plt.tight_layout()
#     plt.show()

# def create_sss_summary_statistics(comp_df):
#     """创建SSS数据集的统计摘要"""
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
#     method = 'With Extra Parsed SQL'
#     method_data = comp_df[comp_df['Method'] == method]
    
#     # 1. 改进/退化统计
#     improved_count = (method_data['Difference'] > 0).sum()
#     degraded_count = (method_data['Difference'] < 0).sum()
#     unchanged_count = (method_data['Difference'] == 0).sum()
    
#     counts = [improved_count, degraded_count, unchanged_count]
#     labels = ['Improved', 'Degraded', 'Unchanged']
#     colors = ['green', 'red', 'gray']
    
#     bars1 = ax1.bar(labels, counts, color=colors, alpha=0.7)
#     ax1.set_title('SSS: Improvement/Degradation Count', fontweight='bold')
#     ax1.set_ylabel('Number of Categories')
#     ax1.grid(axis='y', alpha=0.3)
    
#     # 添加数值标签
#     for bar, count in zip(bars1, counts):
#         ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
#                 f'{count}', ha='center', va='bottom', fontweight='bold')
    
#     # 2. 改进/退化分布
#     improvements = method_data[method_data['Difference'] > 0]['Difference']
#     degradations = method_data[method_data['Difference'] < 0]['Difference']
    
#     avg_improvement = improvements.mean() if len(improvements) > 0 else 0
#     avg_degradation = degradations.mean() if len(degradations) > 0 else 0
    
#     ax2.bar(['Avg Improvement', 'Avg Degradation'], [avg_improvement, avg_degradation],
#            color=['green', 'red'], alpha=0.7)
#     ax2.set_title('SSS: Average Improvement/Degradation', fontweight='bold')
#     ax2.set_ylabel('Average Difference')
#     ax2.grid(axis='y', alpha=0.3)
#     ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
#     # 添加数值标签
#     ax2.text(0, avg_improvement + 0.001, f'{avg_improvement:.4f}', 
#             ha='center', va='bottom', fontweight='bold')
#     ax2.text(1, avg_degradation - 0.001, f'{avg_degradation:.4f}', 
#             ha='center', va='top', fontweight='bold')
    
#     # 3. 分布直方图
#     ax3.hist(method_data['Difference'], bins=8, alpha=0.6, 
#             color=SSS_COMPARISON_COLORS[method]['color'], edgecolor='black')
#     ax3.set_title('SSS: Distribution of Differences', fontweight='bold')
#     ax3.set_xlabel('Difference from Baseline')
#     ax3.set_ylabel('Frequency')
#     ax3.grid(axis='y', alpha=0.3)
#     ax3.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
#     # 4. 性能变化排序
#     sorted_data = method_data.sort_values('Difference')
#     categories_short = [cat.replace('-Related Errors', '') for cat in sorted_data['Category']]
    
#     bars4 = ax4.barh(range(len(categories_short)), sorted_data['Difference'],
#                     color=[SSS_COMPARISON_COLORS[method]['color'] if d >= 0 else 'red' 
#                           for d in sorted_data['Difference']], alpha=0.7)
    
#     ax4.set_title('SSS: Category-wise Performance Change', fontweight='bold')
#     ax4.set_xlabel('Difference from Baseline')
#     ax4.set_yticks(range(len(categories_short)))
#     ax4.set_yticklabels(categories_short, fontsize=9)
#     ax4.grid(axis='x', alpha=0.3)
#     ax4.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
#     plt.suptitle('SSS Dataset - Statistical Summary of Baseline Comparison', 
#                 fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.show()
    
#     return {
#         'improved': improved_count,
#         'degraded': degraded_count,
#         'unchanged': unchanged_count,
#         'avg_improvement': avg_improvement,
#         'avg_degradation': avg_degradation
#     }

# # ==== 主执行函数 ====
# def main():
#     """主函数：生成SSS数据集的baseline对比图表"""
#     print("Loading and processing SSS data...")
#     df, sub_error_to_category, ordered_categories, ordered_sub_errors = load_and_process_sss_data()
    
#     print("Creating SSS baseline comparison chart...")
#     comp_df, ordered_cats, baseline_sorted = create_sss_baseline_comparison_chart(df, ordered_categories)
    
#     print("Creating SSS detailed comparison...")
#     create_sss_detailed_comparison(df, sub_error_to_category, ordered_cats, baseline_sorted)
    
#     print("Creating SSS summary statistics...")
#     stats = create_sss_summary_statistics(comp_df)
    
#     # 打印统计摘要
#     print("\n=== SSS数据集 Baseline对比统计摘要 ===")
#     print("大类baseline性能排序 (从低到高):")
#     for i, (category, score) in enumerate(baseline_sorted.items(), 1):
#         cat_display = category.replace('-Related Errors', '')
#         print(f"{i}. {cat_display}: {score:.3f}")
    
#     print(f"\nSSS数据集改进/退化统计:")
#     print(f"  - 改进类别: {stats['improved']} 个")
#     print(f"  - 退化类别: {stats['degraded']} 个")
#     print(f"  - 无变化类别: {stats['unchanged']} 个")
#     print(f"  - 平均改进幅度: {stats['avg_improvement']:.4f}")
#     print(f"  - 平均退化幅度: {stats['avg_degradation']:.4f}")

# if __name__ == "__main__":
#     main()