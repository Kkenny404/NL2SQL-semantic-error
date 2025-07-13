# import json
# import os
# from datetime import datetime
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# # ==== CONFIG ====
# RESULT_PATH = "results/CoT_Gemini_results_None_20250713_215524.jsonl"  # 你已有的结果路径
# EVAL_SAVE = True  # 是否保存 eval_summary 文件
# # ===============

# # ==== Load Results ====
# preds = []
# labels = []

# with open(RESULT_PATH, "r") as f:
#     for line in f:
#         ex = json.loads(line)
#         if "prediction" in ex and "label" in ex:
#             preds.append(ex["prediction"])
#             labels.append(ex["label"])

# if not preds:
#     raise ValueError("No predictions loaded!")

# # ==== Evaluation ====
# tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
# accuracy = accuracy_score(labels, preds)
# pp = precision_score(labels, preds)
# pr = recall_score(labels, preds)
# npv = tn / (tn + fn) if (tn + fn) > 0 else 0
# nr = tn / (tn + fp) if (tn + fp) > 0 else 0

# # ==== Summary ====
# evaluation_summary = {
#     "result_path": RESULT_PATH,
#     "accuracy": round(accuracy, 4),
#     "precision": round(pp, 4),
#     "recall": round(pr, 4),
#     "negative_precision": round(npv, 4),
#     "negative_recall": round(nr, 4),
#     "true_negative": int(tn),
#     "false_positive": int(fp),
#     "false_negative": int(fn),
#     "true_positive": int(tp),
# }

# # ==== Save ====
# if EVAL_SAVE:
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     eval_path = f"results/eval_summary_from_results_{timestamp}.json"
#     with open(eval_path, "w") as f:
#         json.dump(evaluation_summary, f, indent=2)
#     print(f"\n✅ Evaluation summary saved to: {eval_path}")

# # ==== Print ====
# print("\n===== Evaluation Result =====")
# for k, v in evaluation_summary.items():
#     print(f"{k}: {v}")




# 计算错误类型召回率并添加每个 error_type 的平均 recall
from collections import defaultdict
import json
import os

# 路径配置
SUBSET_PATH = "bug-data/NL2SQL-Bugs-Subset.json"
RESULT_PATH = "results/CoT/CoT_GPT_results_None_20250713_220628.jsonl"
OUTPUT_PATH = "results/CoT/error_GPT.json"

# 加载原始数据和预测结果
with open(SUBSET_PATH, "r") as f:
    raw_data = json.load(f)

with open(RESULT_PATH, "r") as f:
    results = [json.loads(line) for line in f]

stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct_detected": 0}))

# index raw_data by question
question_to_data = {d["question"]: d for d in raw_data}

matched = 0
skipped = 0

for r in results:
    q = r["question"]
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

print(f"✅ Finished. Matched: {matched}, Skipped: {skipped}")
print("📄 Saved to: {OUTPUT_PATH}")
