import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Step 1: Load JSON ===
with open('SSS-data/random_order_ground_truth.json', 'r') as f:
    data = json.load(f)

# === Step 2: Collect (error_type, sub_error_type) where label == False ===
records = []
for item in data:
    if item.get('label') is False:
        for err in item.get('error_types', []):
            records.append({
                'error_type': err['error_type'],
                'sub_error_type': err['sub_error_type']
            })

# === Step 3: Create DataFrame and count combinations ===
df = pd.DataFrame(records)
count_df = df.groupby(['error_type', 'sub_error_type']).size().reset_index(name='count')

# === Step 4: Pivot table for stacked bar plot ===
pivot_df = count_df.pivot(index='error_type', columns='sub_error_type', values='count').fillna(0)

# === Step 5: Improved color assignment ===
# 获取每个error_type对应的sub_error_types
error_type_subs = {}
for _, row in count_df.iterrows():
    error_type = row['error_type']
    sub_error_type = row['sub_error_type']
    if error_type not in error_type_subs:
        error_type_subs[error_type] = []
    if sub_error_type not in error_type_subs[error_type]:
        error_type_subs[error_type].append(sub_error_type)

# 为每个error_type分配基础颜色
error_types = list(error_type_subs.keys())
base_colors = sns.color_palette("Set2", n_colors=len(error_types))

# 为每个sub_error_type分配颜色（同一error_type下的sub_error_types使用相近颜色）
sub_error_to_color = {}
for i, (error_type, sub_errors) in enumerate(error_type_subs.items()):
    if len(sub_errors) == 1:
        # 如果只有一个sub_error_type，直接使用基础颜色
        sub_error_to_color[sub_errors[0]] = base_colors[i]
    else:
        # 如果有多个sub_error_types，生成渐变色
        base_color = base_colors[i]
        # 生成从浅到深的颜色系列
        shades = sns.light_palette(base_color, n_colors=len(sub_errors)+1, reverse=True)[:-1]
        for j, sub_error in enumerate(sub_errors):
            sub_error_to_color[sub_error] = shades[j]

# 按照pivot_df的列顺序获取颜色
colors = [sub_error_to_color[col] for col in pivot_df.columns]

# === Step 6: 创建图表 ===
plt.style.use('default')  # 确保使用默认样式
fig, ax = plt.subplots(figsize=(16, 8))  # 增加宽度为图例留出更多空间

# 绘制堆叠柱状图
pivot_df.plot(
    kind='bar',
    stacked=True,
    color=colors,
    ax=ax,
    width=0.8,
    edgecolor='white',
    linewidth=0.5
)

# === Step 7: 美化图表 ===
ax.set_ylabel("Number of Occurrences", fontsize=12, fontweight='bold')
ax.set_xlabel("Error Type", fontsize=12, fontweight='bold')
ax.set_title("Distribution of Semantic Error Types and Subtypes", 
             fontsize=16, fontweight='bold', pad=20)

# 旋转x轴标签以提高可读性
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# 添加网格线
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# === Step 8: 按error category分组排列图例 ===
handles, labels = ax.get_legend_handles_labels()

# 创建(error_type, sub_error_type, handle)的映射，按error_type分组
legend_groups = {}
for handle, label in zip(handles, labels):
    # 找到这个sub_error_type对应的error_type
    error_type = None
    for et, subs in error_type_subs.items():
        if label in subs:
            error_type = et
            break
    
    if error_type not in legend_groups:
        legend_groups[error_type] = []
    legend_groups[error_type].append((handle, label))

# 按error_type排序，并重新组织handles和labels
sorted_handles = []
sorted_labels = []
for error_type in sorted(legend_groups.keys()):
    # 按sub_error_type名称排序同一组内的项目
    group_items = sorted(legend_groups[error_type], key=lambda x: x[1])
    for handle, label in group_items:
        sorted_handles.append(handle)
        sorted_labels.append(label)

# 计算合适的列数 - 改为竖列排布
n_labels = len(sorted_labels)
ncol = 1  # 使用单列，完全竖直排布

# 将图例放在右侧，使用分组后的顺序，增加距离
legend = ax.legend(
    sorted_handles, sorted_labels,
    title="Sub Error Type\n(grouped by Error Category)",
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    fontsize=9,
    title_fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=True,
    ncol=ncol,
    columnspacing=0.5,  # 列间距
    handletextpad=0.5   # 图例标记和文本间距
)

# 在图例中添加分组分隔线（可选）
# 计算每个error_type组的位置，添加分隔线
current_pos = 0
for i, (error_type, items) in enumerate(sorted(legend_groups.items())):
    if i > 0:  # 不在第一组前添加线
        # 添加分组标题（使用不同的文本样式）
        y_pos = len(sorted_labels) - current_pos - 0.5
        # 这里可以添加分组标识，但matplotlib的legend比较限制，我们用颜色分组已经足够清晰
    current_pos += len(items)

# === Step 9: 添加数据标签（可选，对于较小的值） ===
# 只为总高度较小的柱子添加标签，避免图表过于拥挤
for i, error_type in enumerate(pivot_df.index):
    total_height = pivot_df.loc[error_type].sum()
    if total_height < 100:  # 只为总数小于100的柱子添加标签
        ax.text(i, total_height + 5, f'{int(total_height)}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# === Step 10: 最终布局调整和保存 ===
plt.tight_layout()

# 保存高质量图片
plt.savefig("error_type_distribution_improved.pdf", 
            bbox_inches='tight', dpi=300, facecolor='white')
plt.savefig("error_type_distribution_improved.png", 
            bbox_inches='tight', dpi=300, facecolor='white')

plt.show()

# === 可选：打印分组信息 ===
print(f"Total error records: {len(df)}")
print(f"Number of error types: {len(error_types)}")
print(f"Number of sub error types: {len(pivot_df.columns)}")
print("\nError categories and their sub-types:")
for error_type in sorted(error_type_subs.keys()):
    print(f"\n{error_type}:")
    for sub_error in sorted(error_type_subs[error_type]):
        count = count_df[
            (count_df['error_type'] == error_type) & 
            (count_df['sub_error_type'] == sub_error)
        ]['count'].iloc[0]
        print(f"  - {sub_error}: {count}")
print("\nTop 5 most common error combinations:")
print(count_df.nlargest(5, 'count')[['error_type', 'sub_error_type', 'count']])