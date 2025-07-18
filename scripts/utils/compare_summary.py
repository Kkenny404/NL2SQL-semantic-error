import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# 文件路径
baseline_file = "results/Baseline/Baseline_summary_table.csv"
cot_file = "results/CoT/CoT_summary_table.csv"
duck_reflex_file = "results/Duck_reflex/Duck_reflex_gpt4o_adaptive_None_20250714_070836_basic_summary.json"

# 读取两个表格
baseline_df = pd.read_csv(baseline_file)
cot_df = pd.read_csv(cot_file)

# 合并两个表格以便比较
comparison_df = baseline_df.merge(cot_df, on="Model", suffixes=("_Baseline", "_CoT"))

# 转换为百分数并计算提升
metrics = ["F1 Score", "Positive Precision", "Positive Recall", "Negative Precision", "Negative Recall"]
formatted_data = pd.DataFrame()
formatted_data["Model"] = comparison_df["Model"]

for metric in metrics:
    formatted_data[f"{metric} Baseline"] = (comparison_df[f"{metric}_Baseline"] * 100).round(2).astype(str) + "%"
    formatted_data[f"{metric} CoT"] = (
        (comparison_df[f"{metric}_CoT"] * 100).round(2).astype(str) +
        " (" +
        (comparison_df[f"{metric}_CoT"] - comparison_df[f"{metric}_Baseline"]).mul(100).round(2).astype(str) +
        ")"
    )

# 创建分组表格
output_file = "results/Merged_Comparison_Percentage_Grouped.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    workbook = writer.book
    worksheet = workbook.active
    worksheet.title = "Comparison Results"

    # 添加表头
    headers = ["Model"] + [f"{metric} Baseline" for metric in metrics] + [f"{metric} CoT" for metric in metrics]
    worksheet.append(headers)

    # 添加数据
    for row in dataframe_to_rows(formatted_data, index=False, header=False):
        worksheet.append(row)

print(f"Merged comparison table has been exported to {output_file}")