import os
import json

# 目录路径
sql_folder = "spider2-lite/evaluation_suite/GoldWithError/sql"
jsonl_path = "spider2-lite/spider2-lite.jsonl"
output_jsonl_path = "spider2-lite/spider2-lite_filtered.jsonl"

# 1. 读取SQL文件名，提取instance_id
sql_ids = set()
for filename in os.listdir(sql_folder):
    if filename.endswith(".sql"):
        instance_id = filename[:-4]  # 去掉 .sql
        sql_ids.add(instance_id)

print(f"Found {len(sql_ids)} SQL files.")

# 2. 逐行读取JSONL
filtered_lines = []
with open(jsonl_path, "r", encoding="utf-8") as infile:
    for line in infile:
        obj = json.loads(line)
        if obj["instance_id"] in sql_ids:
            filtered_lines.append(line)

print(f"Remaining {len(filtered_lines)} records after filtering.")

# 3. 输出新的JSONL文件
with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
    outfile.writelines(filtered_lines)

print(f"New JSONL saved to {output_jsonl_path}.")




## ============================
# 下面的代码用于从新的jsonl文件中提取不同类型的数据库列表
# ============================

import json

# 输入你的新的jsonl文件
jsonl_path = output_jsonl_path

# 三个集合
local_dbs = set()
sf_dbs = set()
bq_dbs = set()

# 按行读取
with open(jsonl_path, "r", encoding="utf-8") as infile:
    for line in infile:
        obj = json.loads(line)
        iid = obj["instance_id"]
        db = obj["db"]
        if iid.startswith("local"):
            local_dbs.add(db)
        elif iid.startswith("sf"):
            sf_dbs.add(db)
        elif iid.startswith("bq"):
            bq_dbs.add(db)

# 输出结果
print("=== Local DBs ===")
for db in sorted(local_dbs):
    print(db)

print("\n=== SF DBs ===")
for db in sorted(sf_dbs):
    print(db)

print("\n=== BQ DBs ===")
for db in sorted(bq_dbs):
    print(db)

# 如果需要写入文件
with open("spider2-lite/db_lists.txt", "w", encoding="utf-8") as f:
    f.write("=== Local DBs ===\n")
    for db in sorted(local_dbs):
        f.write(db + "\n")
    f.write("\n=== SF DBs ===\n")
    for db in sorted(sf_dbs):
        f.write(db + "\n")
    f.write("\n=== BQ DBs ===\n")
    for db in sorted(bq_dbs):
        f.write(db + "\n")

print("\nAll db lists saved to spider2-lite/db_lists.txt.")






