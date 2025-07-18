import json
import pandas as pd

# 文件路径
file_paths = {
    "GPT-4o": "results/CoT/CoT_GPT_results_None_20250713_220628_basic_summary.json",
    "Claude-sonnet-4": "results/CoT/CoT_CLAUD_results_None_20250713_214609_basic_summary.json",
    "Gemini-2.5-flash": "results/CoT/CoT_Gemini_results_None_20250714_051846_basic_summary.json"
}

# 存储结果
results = []

# 读取每个文件并提取数据
for model_name, file_path in file_paths.items():
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)  # 解析 JSON 文件

        # 添加到结果列表
        results.append({
            "Model": model_name,
            "Accuracy": round(data["accuracy"], 4),
            "Positive Precision": round(data["positive_precision"], 4),
            "Positive Recall": round(data["positive_recall"], 4),
            "Negative Precision": round(data["negative_precision"], 4),
            "Negative Recall": round(data["negative_recall"], 4),
            "Overall Precision Avg": round(data["overall_precision_avg"], 4),
            "Overall Recall Avg": round(data["overall_recall_avg"], 4),
            "F1 Score": round(data["f1_score"], 4)
        })
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# 创建 DataFrame
df = pd.DataFrame(results)

# 打印表格
print(df)

# 导出到 CSV 文件
output_file = "results/CoT/CoT_summary_table.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Table has been exported to {output_file}")