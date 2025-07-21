# import os
# import random
# import json

# # 定义路径
# sql_directory = "Spider/lite_256_true_sql"
# jsonl_path = "Spider/spider2-lite_with_golden.jsonl"
# output_file = "Spider/sql_with_conditions.json"

# # 筛选包含 JOIN 的 SQL 文件
# def filter_sql_with_join(directory):
#     sql_files_with_join = []
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         if os.path.isfile(file_path) and filename.endswith(".sql"):
#             with open(file_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 if "JOIN" in content.upper():  # 检查是否包含 JOIN
#                     sql_files_with_join.append(filename)
#     return sql_files_with_join

# # 加载 JSONL 文件并创建 instance_id 到 db、question 和 external_knowledge 的映射
# def load_metadata(jsonl_file):
#     instance_id_to_metadata = {}
#     with open(jsonl_file, "r", encoding="utf-8") as f:
#         for line in f:
#             entry = json.loads(line)
#             instance_id = entry.get("instance_id")
#             db = entry.get("db")
#             question = entry.get("question")
#             external_knowledge = entry.get("external_knowledge")
#             if instance_id and db and question:
#                 instance_id_to_metadata[instance_id] = {
#                     "db": db,
#                     "question": question,
#                     "external_knowledge": external_knowledge
#                 }
#     return instance_id_to_metadata

# # 主逻辑
# sql_files_with_join = filter_sql_with_join(sql_directory)
# if len(sql_files_with_join) < 20:
#     print("❌ 文件数量不足 20 个包含 JOIN 的 SQL 文件")
# else:
#     selected_files = random.sample(sql_files_with_join, 20)

#     # 加载 instance_id 到 db、question 和 external_knowledge 的映射
#     instance_id_to_metadata = load_metadata(jsonl_path)

#     # 分成两组并保存到 JSON 文件
#     results = []

#     for name in selected_files[:10]:
#         original_name = os.path.splitext(name)[0]
#         metadata = instance_id_to_metadata.get(original_name, {
#             "db": "❓ DB not found",
#             "question": "❓ Question not found",
#             "external_knowledge": None
#         })
#         results.append({
#             "true_sql_id": original_name,
#             "sql_id": f"{original_name}_JoinTypeMismatch.sql",
#             "db": metadata["db"],
#             "question": metadata["question"],
#             "error_types": [
#                 {
#                     "error_type": "Table-related Errors",
#                     "sub_error_type": "Join Type Mismatch"
#                 }
#             ],
#             "external_knowledge": metadata["external_knowledge"]
#         })

#     for name in selected_files[10:]:
#         original_name = os.path.splitext(name)[0]
#         metadata = instance_id_to_metadata.get(original_name, {
#             "db": "❓ DB not found",
#             "question": "❓ Question not found",
#             "external_knowledge": None
#         })
#         results.append({
#             "true_sql_id": original_name,
#             "sql_id": f"{original_name}_JoinConditionMismatch.sql",
#             "db": metadata["db"],
#             "question": metadata["question"],
#             "error_types": [
#                 {
#                     "error_type": "Table-related Errors",
#                     "sub_error_type": "Join Condition Mismatch"
#                 }
#             ],
#             "external_knowledge": metadata["external_knowledge"]
#         })

#     # 保存到 JSON 文件
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)


#     print(f"✅ Results saved to {output_file}")



import os
import json

# 定义路径
sql_directory = "Spider/Error_data/Table-related Errors"
jsonl_path = "Spider/Error_data/Table-related Errors/ground_truth.json"

# 获取 SQL 文件名（去掉 .sql 后缀）
def get_sql_file_names(directory):
    sql_file_names = set()
    for filename in os.listdir(directory):
        if filename.endswith(".sql"):
            sql_file_names.add(os.path.splitext(filename)[0])
    return sql_file_names

# 加载 JSON 文件中的 instance_id
def get_instance_ids(json_file):
    instance_ids = set()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # 加载 JSON 数组
        for entry in data:
            instance_id = entry.get("sql_id")  # ground_truth.json 使用 true_sql_id
            if instance_id:
                instance_ids.add(os.path.splitext(instance_id)[0])  # 去掉 .sql 后缀
    return instance_ids

# 主逻辑
sql_file_names = get_sql_file_names(sql_directory)
instance_ids = get_instance_ids(jsonl_path)

# 找出没有匹配的文件名
unmatched_file_names = sql_file_names - instance_ids

# 打印结果
if unmatched_file_names:
    print("以下文件名没有匹配到 instance_id：")
    i= 0
    for file_name in unmatched_file_names:
        print(file_name)
        i += 1
    print(f"总共有 {i} 个文件名没有匹配到 instance_id。")
else:
    print("所有文件名都匹配到 instance_id。")


# 找出匹配的文件名
matched_file_names = sql_file_names & instance_ids

# 打印结果
if matched_file_names:
    print("以下文件名匹配到 instance_id：")
    for file_name in matched_file_names:
        print(file_name)
    print(f"总共有 {len(matched_file_names)} 个文件名匹配到 instance_id。")
else:
    print("没有文件名匹配到 instance_id。")

# 删除未匹配的文件
# if unmatched_file_names:
#     print("以下文件名没有匹配到 instance_id，将被删除：")
#     for file_name in unmatched_file_names:
#         file_path = os.path.join(sql_directory, f"{file_name}.sql")
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"Deleted: {file_path}")
# else:
#     print("所有文件名都匹配到 instance_id，无需删除。")