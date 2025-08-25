import json

def compute_f1(precision, recall):
    return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def write_f1_back_to_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    data["pos_f1"] = compute_f1(data.get("precision", 0), data.get("recall", 0))
    data["neg_f1"] = compute_f1(data.get("negative_precision", 0), data.get("negative_recall", 0))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated file: {file_path}")

# Example
write_f1_back_to_file("results/eval_result/4o.json")
