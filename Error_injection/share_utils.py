import json
import random
import os
import re
from pathlib import Path
import sys
import glob
from abc import ABC, abstractmethod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.sqlite_schema_extract import sqlite_schema_extract_format


# ========== 基础工具函数 ==========

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

def load_sql(sql_root, sql_id):
    """Load the SQL content from a file based on the given SQL ID."""
    sql_path = os.path.join(sql_root, f"{sql_id}.sql")
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    with open(sql_path, "r", encoding="utf-8") as f:
        return f.read()

def get_schema(instance_id, db_id, schema_root):
    if instance_id.startswith("sf"):
        base = os.path.join(schema_root, "snowflake", db_id)
    elif instance_id.startswith("bq") or instance_id.startswith("ga"):
        base = os.path.join(schema_root, "bigquery", db_id)
    elif instance_id.startswith("local"):
        sqlite_path = os.path.join(schema_root, "spider2-localdb", f"{db_id}.sqlite")
        print(f"Trying to open SQLite file at: {sqlite_path}")
        return sqlite_schema_extract_format(sqlite_path)
    else:
        raise ValueError("❌ 无法识别 instance_id 的前缀")

    # 查找 DDL.csv 文件
    ddl_paths = []
    direct_path = os.path.join(base, "DDL.csv")
    if os.path.exists(direct_path):
        ddl_paths.append(direct_path)

    sub_paths = glob.glob(os.path.join(base, "**", "DDL.csv"), recursive=True)
    for path in sub_paths:
        if path not in ddl_paths:
            ddl_paths.append(path)

    if not ddl_paths:
        raise FileNotFoundError(f"DDL.csv not found in {base} or its subdirectories")

    all_ddl = []
    for path in ddl_paths:
        with open(path, "r", encoding="utf-8") as f:
            all_ddl.append(f.read())

    return "\n\n".join(all_ddl)

def schema_shrink(schema: str) -> str:
    """对输入的数据库 schema 进行去重和简化处理"""
    table_pattern = re.compile(r"CREATE TABLE `(.+?)`\s*\((.*?)\);", re.DOTALL)
    field_pattern = re.compile(r"(\w+)\s+(\w+.*?)(,|$)")

    schema_map = {}

    for table_match in table_pattern.finditer(schema):
        table_name = table_match.group(1).strip()
        fields = table_match.group(2)
        field_list = []
        for field_match in field_pattern.finditer(fields):
            field_name = field_match.group(1).strip()
            field_type = field_match.group(2).strip()
            field_list.append(f"{field_name} {field_type}")
        field_tuple = tuple(sorted(field_list))
        if field_tuple not in schema_map:
            schema_map[field_tuple] = [table_name]
        else:
            schema_map[field_tuple].append(table_name)

    simplified_schema = []
    for fields, tables in schema_map.items():
        representative_table = tables[0]
        simplified_schema.append(f"CREATE TABLE `{representative_table}` (\n    " + ",\n    ".join(fields) + "\n);")
        if len(tables) > 1:
            simplified_schema.append(f"-- Tables with the same structure: {', '.join(tables[1:])}")

    return "\n\n".join(simplified_schema)





# ===================== prompt builder ===================================
def build_prompt(question: str,
        correct_sql: str,
        schema: str,
        category: str,
        category_description: str,
        sub_error_type: str,
        error_description: str) -> str:
    """构造用于生成包含特定语义错误的 SQL 的 prompt"""
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

