import json
from typing import OrderedDict

input_path = "Spider/spider2-lite_with_golden.jsonl"   # 你的输入文件
output_path = "Spider/spider2-lite_formatted.jsonl" # 输出文件

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():
            continue
        data = json.loads(line)
        # 构造新的 OrderedDict 保证顺序
        ordered_data = OrderedDict()
        ordered_data["sql_id"] = data.pop("instance_id") + ".sql"
        ordered_data["db"] = data.get("db")
        ordered_data["question"] = data.get("question")
        ordered_data["external_knowledge"] = data.get("external_knowledge")
        ordered_data["error_types"] = []
        ordered_data["label"] = True
        # 写入
        outfile.write(json.dumps(ordered_data, ensure_ascii=False) + "\n")

print(f"转换完成！输出文件：{output_path}")

