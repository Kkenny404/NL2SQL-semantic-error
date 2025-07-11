import json
import pandas as pd

# 替换为你的文件路径
file_paths = {
    "Gemini": "results/eval_GEMINIsummary_1199_20250711_031233.json",
    "GPT": "results/eval_summary_from_results_20250710_202503.json"
}

# 存储结果
results = []

for model_name, file_path in file_paths.items():
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()  # Read the entire file content
            data = json.loads(file_content)  # Parse the full JSON object

        tp = data["true_positive"]
        fp = data["false_positive"]
        fn = data["false_negative"]
        tn = data["true_negative"]

        # 计算指标
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Precision
        nr = tn / (tn + fp) if (tn + fp) > 0 else 0   # Negative Recall
        pp = tp / (tp + fp) if (tp + fp) > 0 else 0   # Positive Precision
        pr = tp / (tp + fn) if (tp + fn) > 0 else 0   # Positive Recall

        results.append({
            "Model": model_name,
            "Negative Precision (NP)": round(npv, 4),
            "Negative Recall (NR)": round(nr, 4),
            "Positive Precision (PP)": round(pp, 4),
            "Positive Recall (PR)": round(pr, 4),
        })
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
    except KeyError as e:
        print(f"Missing key in JSON data from file {file_path}: {e}")

# 创建 DataFrame 并输出
df = pd.DataFrame(results)
print(df)

# 导出到 CSV 文件
output_file = "results/table_summary.csv"  # 指定输出文件路径
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"Results have been exported to {output_file}")