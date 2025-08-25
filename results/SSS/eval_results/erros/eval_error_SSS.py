# 计算错误类型召回率并添加每个 error_type 的平均 recall
from collections import defaultdict
import json
import os

# 路径配置
SUBSET_PATH = "SSS-data/random_order_ground_truth.json"
RESULT_PATH = "results/SSS/Baseline_None/GPT4o_20250813_161210.jsonl"
OUTPUT_PATH = "results/SSS/eval_results/erros/4o_error.json"

# 加载原始数据和预测结果
with open(SUBSET_PATH, "r") as f:
    raw_data = json.load(f)

# with open(RESULT_PATH, "r") as f:
#     results = [json.loads(line) for line in f]

results = []
with open(RESULT_PATH, "r") as f:
    for line in f:
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[WARN] 无法解析的行: {line.strip()}")

stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct_detected": 0}))

# index raw_data by question
question_to_data = {d["sql_id"]: d for d in raw_data}

matched = 0
skipped = 0

for r in results:
    q = r["sql_id"]
    d = question_to_data.get(q)
    if d is None:
        skipped += 1
        continue

    matched += 1
    label = d["label"]
    pred = r["prediction"]
    errors = d.get("error_types", [])

    for err in errors:
        main_type = err["error_type"]
        sub_type = err["sub_error_type"]
        stats[main_type][sub_type]["total"] += 1
        if label is False and pred is False:
            stats[main_type][sub_type]["correct_detected"] += 1


# 计算 recall 并格式化输出
result_summary = {}
for main_type, sub_dict in stats.items():
    result_summary[main_type] = {}
    recalls = []
    for sub_type, values in sub_dict.items():
        total = values["total"]
        correct = values["correct_detected"]
        recall = correct / total if total > 0 else 0
        recalls.append(recall)
        result_summary[main_type][sub_type] = {
            "total": total,
            "detected_false_positives": correct,
            "recall": round(recall, 4)
        }
    # 添加平均 recall
    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        result_summary[main_type]["average_recall"] = round(avg_recall, 4)

# 保存到 JSON 文件
with open(OUTPUT_PATH, "w") as f:
    json.dump(result_summary, f, indent=2)


print(len(results), "results loaded")
print(len(raw_data), "raw data loaded")
print(f"✅ Finished. Matched: {matched}, Skipped: {skipped}")
print("📄 Saved to: {OUTPUT_PATH}")
