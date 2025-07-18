import json
import pandas as pd

# 文件路径
baseline_file = "results/Baseline/error_GPT.json"
cot_file = "results/CoT/error_GPT.json"
cot_reflex_file = "results/Duck_reflex/error_GPT4o.json"  # 新增文件
output_file = "results/Comparison_error_GPT.xlsx"

# 读取三个 JSON 文件
with open(baseline_file, "r") as f:
    baseline_data = json.load(f)

with open(cot_file, "r") as f:
    cot_data = json.load(f)

with open(cot_reflex_file, "r") as f:
    cot_reflex_data = json.load(f)

# 对比提升与下降
comparison_results = {}
for error_category in baseline_data.keys():
    if error_category in cot_data and error_category in cot_reflex_data:
        comparison_results[error_category] = {}
        for error_type in baseline_data[error_category].keys():
            if error_type in cot_data[error_category] and error_type in cot_reflex_data[error_category]:
                baseline_value = baseline_data[error_category][error_type]
                cot_value = cot_data[error_category][error_type]
                cot_reflex_value = cot_reflex_data[error_category][error_type]

                # 如果值是字典类型，提取 "recall"；否则直接比较
                if isinstance(baseline_value, dict) and isinstance(cot_value, dict) and isinstance(cot_reflex_value, dict):
                    baseline_recall = baseline_value.get("recall", None)
                    cot_recall = cot_value.get("recall", None)
                    cot_reflex_recall = cot_reflex_value.get("recall", None)
                else:
                    baseline_recall = baseline_value
                    cot_recall = cot_value
                    cot_reflex_recall = cot_reflex_value

                # 计算差异
                if baseline_recall is not None and cot_recall is not None and cot_reflex_recall is not None:
                    comparison_results[error_category][error_type] = {
                        "Baseline Recall": baseline_recall * 100,  # 转换为百分数
                        "CoT Recall": cot_recall * 100,  # 转换为百分数
                        "CoT Reflex Recall": cot_reflex_recall * 100,  # 转换为百分数
                        "Difference (CoT - Baseline)": (cot_recall - baseline_recall) * 100,  # 转换为百分数
                        "Difference (CoT Reflex - Baseline)": (cot_reflex_recall - baseline_recall) * 100,  # 转换为百分数
                        "Difference (CoT Reflex - CoT)": (cot_reflex_recall - cot_recall) * 100  # 转换为百分数
                    }

# 计算每个类别的平均 recall
for category, errors in comparison_results.items():
    baseline_avg = sum(values["Baseline Recall"] for values in errors.values()) / len(errors)
    cot_avg = sum(values["CoT Recall"] for values in errors.values()) / len(errors)
    cot_reflex_avg = sum(values["CoT Reflex Recall"] for values in errors.values()) / len(errors)
    comparison_results[category]["average_recall"] = {
        "Baseline Recall": baseline_avg,
        "CoT Recall": cot_avg,
        "CoT Reflex Recall": cot_reflex_avg,
        "Difference (CoT - Baseline)": cot_avg - baseline_avg,
        "Difference (CoT Reflex - Baseline)": cot_reflex_avg - baseline_avg,
        "Difference (CoT Reflex - CoT)": cot_reflex_avg - cot_avg
    }

# 转换为 DataFrame
rows = []
for category, errors in comparison_results.items():
    for error_type, values in errors.items():
        if error_type != "average_recall":  # 排除 average_recall，稍后单独处理
            rows.append({
                "Category": category,
                "Error Type": error_type,
                "Baseline Recall": f"{values['Baseline Recall']:.2f}%",  # 转换为百分数格式
                "CoT Recall": f"{values['CoT Recall']:.2f}%",  # 转换为百分数格式
                "CoT Reflex Recall": f"{values['CoT Reflex Recall']:.2f}%",  # 转换为百分数格式
                "Difference (CoT - Baseline)": f"{values['Difference (CoT - Baseline)']:+.2f}%",  # 转换为百分数格式
                "Difference (CoT Reflex - Baseline)": f"{values['Difference (CoT Reflex - Baseline)']:+.2f}%",  # 转换为百分数格式
                "Difference (CoT Reflex - CoT)": f"{values['Difference (CoT Reflex - CoT)']:+.2f}%"  # 转换为百分数格式
            })
    # 插入 average recall 到每个类别的下一行
    avg = errors["average_recall"]
    rows.append({
        "Category": category,
        "Error Type": "Average Recall",
        "Baseline Recall": f"{avg['Baseline Recall']:.2f}%",
        "CoT Recall": f"{avg['CoT Recall']:.2f}%",
        "CoT Reflex Recall": f"{avg['CoT Reflex Recall']:.2f}%",
        "Difference (CoT - Baseline)": f"{avg['Difference (CoT - Baseline)']:+.2f}%",
        "Difference (CoT Reflex - Baseline)": f"{avg['Difference (CoT Reflex - Baseline)']:+.2f}%",
        "Difference (CoT Reflex - CoT)": f"{avg['Difference (CoT Reflex - CoT)']:+.2f}%"
    })

df = pd.DataFrame(rows)

# 导出为 Excel 文件
df.to_excel(output_file, index=False, sheet_name="Comparison Results")
print(f"Comparison results have been exported to {output_file}")