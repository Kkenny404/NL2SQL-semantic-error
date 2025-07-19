# import json

# # 输入输出路径
# # input_path = "Spider/9Errors/spider2-lite_Attribute-related Errors_errors.jsonl"
# # output_path = "Spider/9Errors/spider2-lite_Attribute-related Errors_errors.json"

PATH = "Spider/Error_data/Table-related Errors/ground_truth.jsonl"
output_path = "Spider/Error_data/Table-related Errors/ground_truth.json"

# # 读取每一行 JSONL 并组装成列表
# with open(PATH, "r", encoding="utf-8") as f:
#     data = [json.loads(line.strip()) for line in f if line.strip()]

# # 写入标准 JSON 格式（数组形式）
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)

# print(f"✅ 已成功转换为 JSON 格式，输出文件: {output_path}")


import json

def add_false_label_to_jsonl(input_path, output_path):
    data = []
    # 逐行读取每一条 JSON 记录
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item["label"] = False
            data.append(item)
    
    # 保存为 JSON 数组（可选也可继续保存为 .jsonl）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(data)} JSONL entries to JSON with label=false → saved to: {output_path}")

# 示例调用（你可以改成自己的文件名）
add_false_label_to_jsonl(PATH, output_path)
