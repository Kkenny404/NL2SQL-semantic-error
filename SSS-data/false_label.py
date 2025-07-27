# import json

# input_path = "SSS-data/merged_ground_truth.jsonl"    # 输入文件
# output_path = "SSS-data/false.jsonl"  # 输出文件

# with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
#     for line in infile:
#         if not line.strip():
#             continue
#         data = json.loads(line)
#         data["label"] = False  # 新增或覆盖
#         outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

# print(f"已处理完成，输出文件：{output_path}")


import json
import random

input_path = "SSS-data/ground_truth.jsonl"
output_path = "SSS-data/random_order_ground_truth.jsonl"

# 1. 读取所有行
with open(input_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

# 2. 打乱顺序
random.shuffle(lines)

# 3. 写回文件
with open(output_path, "w", encoding="utf-8") as f:
    for obj in lines:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"已打乱顺序并写入 {output_path}")