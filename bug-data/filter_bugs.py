import json
import random

# 路径设置
input_path = "bug-data/NL2SQL-Bugs.json"
output_path = "bug-data/NL2SQL-Bugs-Subset.json"


# 设置随机种子（确保可复现）
random.seed(42)

# 读取数据
with open(input_path, "r") as f:
    data = json.load(f)

# 分成正确和错误的
correct = [item for item in data if item["label"] == True]
incorrect = [item for item in data if item["label"] == False]

# 随机选取200个正确
selected_correct = random.sample(correct, 200)

# 合并成新的数据集
subset = incorrect + selected_correct

# 打乱顺序
random.shuffle(subset)

# 保存
with open(output_path, "w") as f:
    json.dump(subset, f, indent=2, ensure_ascii=False)

print(f"总样本数: {len(subset)}（错误: {len(incorrect)}，正确: {len(selected_correct)}）")
print(f"已保存到: {output_path}")
