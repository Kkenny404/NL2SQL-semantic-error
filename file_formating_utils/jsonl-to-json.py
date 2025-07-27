import json

# 输入输出路径
# input_path = "Spider/9Errors/spider2-lite_Attribute-related Errors_errors.jsonl"
# output_path = "Spider/9Errors/spider2-lite_Attribute-related Errors_errors.json"

PATH = "SSS-data/random_order_ground_truth.jsonl"
output_path = "SSS-data/random_order_ground_truth.json"

# 读取每一行 JSONL 并组装成列表
with open(PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# 写入标准 JSON 格式（数组形式）
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ 已成功转换为 JSON 格式，输出文件: {output_path}")
