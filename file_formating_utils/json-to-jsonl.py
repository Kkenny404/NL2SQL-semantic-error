import json

# 输入输出路径
input_path = "Error_injection/Error_data/Value-related Errors/ground_truth.json"
output_path = "Error_injection/Error_data/Value-related Errors/ground_truth.jsonl"

# 读取 JSON 文件
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 加载 JSON 数组

# 写入 JSONL 文件
with open(output_path, "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ 已成功转换为 JSONL 格式，输出文件: {output_path}")