import json
import random
import os
import re
from pathlib import Path
import sys
import glob
from typing import Callable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.sqlite_schema_extract import sqlite_schema_extract_format


# def build_prompt(question: str, schema: str, correct_sql: str,
#                  error_type: str, sub_error_type: str, error_description: str) -> str:
#     """
#     构造用于生成包含特定语义错误的 SQL 的 prompt。
#     """
#     prompt = f"""You are an expert at injecting SQL semantic errors.

# Given a natural language question, database schema, and its correct SQL query, please change the correct SQL to include the specific semantic error described below.
# The meaning of this {error_type} -- {sub_error_type} is as follows:
# {error_description}

# The new SQL should include only this error, and no other types of errors. Maintain valid syntax and keep the structure as close as possible to the original.

# Return only the incorrect SQL query.

# ---

# Question:
# {question}

# Database Schema:
# {schema}

# Correct SQL:
# {correct_sql}
# """
#     return prompt


def get_sub_error_types(error_data, category_name):
    for category in error_data:
        if category["category"] == category_name:
            return category["sub_errors"]
    return []

def load_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_error_explanation(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_schema(instance_id, db_id, schema_root):
    if instance_id.startswith("sf"):
        base = os.path.join(schema_root, "snowflake", db_id)
    elif instance_id.startswith("bq") or instance_id.startswith("ga"):
        base = os.path.join(schema_root, "bigquery", db_id)
    elif instance_id.startswith("local"):
        sqlite_path = os.path.join(schema_root, "local_sqlite", f"{db_id}.sqlite")
        return sqlite_schema_extract_format(sqlite_path)
    else:
        raise ValueError("❌ 无法识别 instance_id 的前缀")

    # ✅ 查找 base 目录下自己的 DDL.csv
    ddl_paths = []
    direct_path = os.path.join(base, "DDL.csv")
    if os.path.exists(direct_path):
        ddl_paths.append(direct_path)

    # ✅ 递归查找所有子目录里的 DDL.csv
    sub_paths = glob.glob(os.path.join(base, "**", "DDL.csv"), recursive=True)
    for path in sub_paths:
        if path not in ddl_paths:  # 避免重复
            ddl_paths.append(path)

    if not ddl_paths:
        raise FileNotFoundError(f"DDL.csv not found in {base} or its subdirectories")

    # ✅ 合并所有 DDL 内容
    all_ddl = []
    for path in ddl_paths:
        with open(path, "r", encoding="utf-8") as f:
            all_ddl.append(f.read())

    return "\n\n".join(all_ddl)

# def build_prompt(question, sql, schema, sub_error_description,):
    # return (
    #     f"You are an expert at injecting SQL semantic errors.\n\n"
    #     f"Given the following natural language question:\n{question}\n\n"
    #     f"And the corresponding correct SQL query:\n{sql}\n\n"
    #     f"And the database schema:\n{schema}\n\n"
    #     f"Please inject an error for the following type:\n{sub_error_description}\n\n"
    #     f"Return the modified SQL only."
    # )
def build_prompt(question: str,
        correct_sql: str,
        schema: str,
        category: str,
        category_description: str,
        sub_error_type: str,
        error_description: str) -> str:
    """
    构造用于生成包含特定语义错误的 SQL 的 prompt。
    """
    prompt = f"""You are an expert at injecting SQL semantic errors.

Given a natural language question, database schema, and its correct SQL query, please change the correct SQL to include the specific semantic error described below.
This error called {sub_error_type} in the category of {category} -- {category_description}.
The meaning of this error is as follows:
{error_description}

The new SQL should include only this error, and no other types of errors. Maintain valid syntax and keep the structure as close as possible to the original.

Return only the incorrect SQL query.

---

Question:
{question}

Database Schema:
{schema}

Correct SQL:
{correct_sql}
"""
    return prompt

def inject_errors_for_category(
    category_name: str,
    error_explanation_path: str,
    jsonl_path: str,
    schema_root: str,
    sql_root: str,
    output_dir: str,
    # llm_call_function: Callable[[str], str]
):
    os.makedirs(f"{output_dir}/{category_name}", exist_ok=True)
    error_data = load_error_explanation(error_explanation_path)
    dataset = load_jsonl(jsonl_path)
    sub_errors = get_sub_error_types(error_data, category_name)
    ground_truth_records = []

    for sub_error in sub_errors:
        sampled_data = random.sample(dataset, k=10)
        for item in sampled_data:
            try:
                instance_id = item["instance_id"]
                db = item["db"]
                question = item["question"]
                external_knowledge = item.get("external_knowledge", "")
                true_sql_path = os.path.join(sql_root, f"{instance_id}.sql")
                if not os.path.exists(true_sql_path):
                    continue
                true_sql = read_file(true_sql_path)
                # schema = get_schema(instance_id, db, schema_root)
                # prompt = build_prompt(question, true_sql, schema, sub_error["description"])
                # error_sql = llm_call_function(prompt)
                
                error_sql = true_sql  # Placeholder for LLM call, replace with actual LLM call
                error_sql_filename = f"{instance_id}_{sub_error['error_type'].replace(' ', '')}.sql"
                error_sql_path = os.path.join(output_dir, category_name, error_sql_filename)
                with open(error_sql_path, "w", encoding="utf-8") as f:
                    f.write(error_sql.strip())

                ground_truth_records.append({
                    "true_sql_id": instance_id,
                    "sql_id": error_sql_filename,
                    "db": db,
                    "question": question,
                    "error_types": [
                        {
                            "error_type": category_name,
                            "sub_error_type": sub_error["error_type"]
                        }
                    ],
                    "external_knowledge": external_knowledge
                })
            except Exception as e:
                print(f"⚠️ Error processing {item['instance_id']}: {e}")

    # Save ground truth
    ground_truth_path = os.path.join(output_dir, category_name, "ground_truth.jsonl")
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        for record in ground_truth_records:
            f.write(json.dumps(record) + "\n")



error_explanation_path = "Spider/error_explaination.json"
output_path = "Spider/Error_data/Table-related Errors"
def generate_prompt_from_groundtruth(
    error_explanation_path: str,
    category: str,
    sub_error_type: str,
    sql_id: str,
    groundtruth_path: str,
    schema_root: str,
    sql_root: str,
    output_path: str
):

    # 1. 加载错误解释
    with open(error_explanation_path, "r", encoding="utf-8") as f:
        error_data = json.load(f)

    category_description, sub_error_description = None, None
    for cat in error_data:
        if cat["category"] == category:
            category_description = cat["description"]
            for sub in cat["sub_errors"]:
                if sub["error_type"] == sub_error_type:
                    sub_error_description = sub["description"]
                    break
    if not category_description or not sub_error_description:
        raise ValueError("❌ 无法在解释文件中找到对应的 category 或 sub_error_type")

    # 2. 从 groundtruth.jsonl 中找 question 和 db
    matched_entry = None
    with open(groundtruth_path, "r", encoding="utf-8") as f:
        records = json.load(f)  # 注意不是 json.loads()，因为你加载的是整个数组
        for record in records:
            if record["true_sql_id"] == sql_id:
                matched_entry = record
                break
    if not matched_entry:
        raise ValueError(f"❌ 未在 ground truth 中找到 true_sql_id = {sql_id}")

    question = matched_entry["question"]
    db_id = matched_entry["db"]
    instance_id = sql_id  # 统一命名
    external_knowledge = matched_entry.get("external_knowledge", "")

    # 3. 加载 schema
    schema = get_schema(instance_id, db_id, schema_root)

    # 4. 加载正确 SQL
    sql_path = os.path.join(sql_root, f"{instance_id}.sql")
    with open(sql_path, "r", encoding="utf-8") as f:
        correct_sql = f.read()

    # 5. 构造 prompt
    prompt = build_prompt(
        question=question,
        correct_sql=correct_sql,
        schema=schema,
        category=category,
        category_description=category_description,
        sub_error_type=sub_error_type,
        error_description=sub_error_description
    )

    # 6. 保存 prompt 文件
    os.makedirs(output_path, exist_ok=True)
    prompt_filename = f"{instance_id}_{sub_error_type.replace(' ', '')}_prompt.txt"
    with open(os.path.join(output_path, prompt_filename), "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"✅ Prompt saved to {os.path.join(output_path, prompt_filename)}")



if __name__ == "__main__":
    category = "Table-related Errors"
    error_explanation_path = "Spider/error_explaination.json"
    jsonl_path = "Spider/spider2-lite_with_golden.jsonl"
    schema_root = "spider2-lite/resource/databases"
    sql_root = "Spider/lite_256_true_sql"
    output_dir = "Spider/Error_data"

    # inject_errors_for_category(
    #     category_name=category,
    #     error_explanation_path=error_explanation_path,
    #     jsonl_path=jsonl_path,
    #     schema_root=schema_root,
    #     sql_root=sql_root,
    #     output_dir=output_dir
    # )

    groundtruth_path = "Spider/Error_data/Table-related Errors/ground_truth.json"
    sub_error_type = "Table Missing"
    sql_id = "bq398"                           
    generate_prompt_from_groundtruth(
        error_explanation_path=error_explanation_path,
        category=category,
        sub_error_type=sub_error_type,
        sql_id=sql_id,
        groundtruth_path=groundtruth_path,
        schema_root=schema_root,
        sql_root=sql_root,
        output_path=output_dir
    )
