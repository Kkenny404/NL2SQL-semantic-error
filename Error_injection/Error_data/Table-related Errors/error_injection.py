import os
import json
import random

def load_groundtruth(groundtruth_path):
    """加载 ground_truth.json 文件"""
    with open(groundtruth_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sql(sql_root, sql_id):
    """加载指定的 SQL 文件"""
    sql_path = os.path.join(sql_root, sql_id)
    with open(sql_path, "r", encoding="utf-8") as f:
        return f.read()

def filter_join_sql(groundtruth, sql_root):
    """筛选包含 JOIN 的 SQL"""
    join_sql_entries = []
    for entry in groundtruth:
        sql_id = entry["sql_id"]
        sql_content = load_sql(sql_root, sql_id)
        if "JOIN" in sql_content.upper():  # 检查是否包含 JOIN
            join_sql_entries.append(entry)
    return join_sql_entries

def inject_errors_for_category(
    category_name,
    error_explanation_path,
    jsonl_path,
    schema_root,
    sql_root,
    output_dir
):
    """注入错误到指定类别"""
    # 加载 ground_truth 数据
    groundtruth_path = os.path.join(output_dir, category_name, "ground_truth.json")
    groundtruth = load_groundtruth(groundtruth_path)

    # 筛选包含 JOIN 的 SQL
    join_sql_entries = filter_join_sql(groundtruth, sql_root)

    # 加载错误解释
    with open(error_explanation_path, "r", encoding="utf-8") as f:
        error_explanations = json.load(f)

    # 遍历 ground_truth 数据
    for entry in groundtruth:
        error_types = entry["error_types"]
        for error in error_types:
            error_type = error["error_type"]
            sub_error_type = error["sub_error_type"]

            # 对于 Join Condition Mismatch 和 Join Type Mismatch，只从包含 JOIN 的 SQL 中随机挑选
            if sub_error_type in ["Join Condition Mismatch", "Join Type Mismatch"]:
                if not join_sql_entries:
                    print(f"⚠️ No JOIN SQL entries found for {sub_error_type}. Skipping...")
                    continue
                entry = random.choice(join_sql_entries)

            # 加载 SQL 和 Schema
            sql_id = entry["sql_id"]
            db_id = entry["db"]
            question = entry["question"]
            schema = get_schema(sql_id, db_id, schema_root)
            correct_sql = load_sql(sql_root, sql_id)

            # 构造错误解释
            category_description = error_explanations[category_name]["description"]
            sub_error_description = error_explanations[category_name]["sub_errors"][sub_error_type]

            # 构造 prompt
            prompt = build_prompt(
                question=question,
                correct_sql=correct_sql,
                schema=schema,
                category=category_name,
                category_description=category_description,
                sub_error_type=sub_error_type,
                error_description=sub_error_description
            )

            # 保存 prompt 文件
            os.makedirs(output_dir, exist_ok=True)
            prompt_filename = f"{sql_id}_{sub_error_type.replace(' ', '')}_prompt.txt"
            with open(os.path.join(output_dir, prompt_filename), "w", encoding="utf-8") as f:
                f.write(prompt)

            print(f"✅ Prompt saved to {os.path.join(output_dir, prompt_filename)}")

if __name__ == "__main__":
    category = "Table-related Errors"
    error_explanation_path = "Spider/error_explaination.json"
    jsonl_path = "Spider/spider2-lite_with_golden.jsonl"
    schema_root = "spider2-lite/resource/databases"
    sql_root = "Spider/lite_256_true_sql"
    output_dir = "Spider/Error_data"

    inject_errors_for_category(
        category_name=category,
        error_explanation_path=error_explanation_path,
        jsonl_path=jsonl_path,
        schema_root=schema_root,
        sql_root=sql_root,
        output_dir=output_dir
    )