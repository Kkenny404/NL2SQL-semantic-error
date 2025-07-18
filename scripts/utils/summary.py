import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# === 修改为你的 JSONL 文件路径 ===
jsonl_path = "results/Duck_reflex/Duck_reflex_gpt4o_adaptive_None_20250714_070836.jsonl"
output_path = os.path.splitext(jsonl_path)[0] + "_basic_summary.json"

# === 读取数据 ===
labels = []
predictions = []

with open(jsonl_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if "label" in data and "prediction" in data:
            labels.append(data["label"])
            predictions.append(data["prediction"])

# === 计算基本指标 ===
accuracy = accuracy_score(labels, predictions)
precision_pos = precision_score(labels, predictions, pos_label=True)
recall_pos = recall_score(labels, predictions, pos_label=True)
f1 = f1_score(labels, predictions, pos_label=True)

# === 混淆矩阵提取 TP/TN/FP/FN ===
tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

# === Negative precision & recall ===
precision_neg = tn / (tn + fn) if (tn + fn) else 0
recall_neg = tn / (tn + fp) if (tn + fp) else 0

# === Overall Precision & Recall ===
overall_precision = (precision_pos + precision_neg) / 2
overall_recall = (recall_pos + recall_neg) / 2

# === 保存为 JSON 文件 ===
summary = {
    "accuracy": accuracy,
    "positive_precision": precision_pos,
    "positive_recall": recall_pos,
    "negative_precision": precision_neg,
    "negative_recall": recall_neg,
    "overall_precision_avg": overall_precision,
    "overall_recall_avg": overall_recall,
    "f1_score": f1
}

with open(output_path, "w") as out_file:
    json.dump(summary, out_file, indent=4)

print(f"Saved evaluation results to {output_path}")
