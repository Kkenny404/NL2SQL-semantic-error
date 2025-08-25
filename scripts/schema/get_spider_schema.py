import glob
import os
import re
import sqlite3


def extract_schema_from_sqlite(sqlite_path: str) -> str:
    """Extract table and column info from .sqlite database"""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        cols = [col[1] for col in cursor.fetchall()]
        schema_parts.append(f"{table}({', '.join(cols)})")
    conn.close()
    return " | ".join(schema_parts)

def get_schema(instance_id, db_id, schema_root, include_all_related=True):
    """
    获取数据库schema，支持包含所有相关的子数据库
    
    Args:
        instance_id: 实例ID
        db_id: 数据库ID  
        schema_root: schema根目录
        include_all_related: 是否包含所有相关的子数据库schema
    """
    if instance_id.startswith("sf"):
        base = os.path.join(schema_root, "snowflake", db_id)
    elif instance_id.startswith("bq") or instance_id.startswith("ga"):
        base = os.path.join(schema_root, "bigquery", db_id)
    elif instance_id.startswith("local"):
        sqlite_path = os.path.join(schema_root, "local_sqlite", f"{db_id}.sqlite")
        print(f"Trying to open SQLite file at: {sqlite_path}")
        return extract_schema_from_sqlite(sqlite_path)
    else:
        raise ValueError("❌ 无法识别 instance_id 的前缀")

    if not os.path.exists(base):
        raise FileNotFoundError(f"Database directory not found: {base}")

    # 查找所有DDL.csv文件
    ddl_paths = []
    schema_info = []
    
    if include_all_related:
        # 方法1: 递归查找所有DDL.csv文件
        all_ddl_files = glob.glob(os.path.join(base, "**", "DDL.csv"), recursive=True)
        
        # 也检查根目录
        root_ddl = os.path.join(base, "DDL.csv")
        if os.path.exists(root_ddl) and root_ddl not in all_ddl_files:
            all_ddl_files.append(root_ddl)
        
        if not all_ddl_files:
            # 如果没有DDL.csv文件，尝试查找其他schema文件
            json_files = glob.glob(os.path.join(base, "**", "*.json"), recursive=True)
            if json_files:
                print(f"⚠️ No DDL.csv found, but found {len(json_files)} JSON files")
                # 可以选择处理JSON文件或者返回空schema
                return "-- No DDL.csv files found, only JSON metadata available"
            raise FileNotFoundError(f"No DDL.csv files found in {base} or its subdirectories")
        
        print(f"📊 Found {len(all_ddl_files)} DDL files for {db_id}")
        
        # 按路径排序，确保一致的顺序
        all_ddl_files.sort()
        
        for ddl_path in all_ddl_files:
            try:
                with open(ddl_path, "r", encoding="utf-8") as f:
                    ddl_content = f.read().strip()
                    if ddl_content:  # 只添加非空内容
                        # 添加数据库路径信息作为注释
                        relative_path = os.path.relpath(ddl_path, base)
                        schema_section = f"-- Schema from: {relative_path}\n{ddl_content}"
                        schema_info.append(schema_section)
            except Exception as e:
                print(f"⚠️ Error reading {ddl_path}: {e}")
                continue
    
    else:
        # 方法2: 只查找直接的DDL.csv文件
        direct_path = os.path.join(base, "DDL.csv")
        if os.path.exists(direct_path):
            with open(direct_path, "r", encoding="utf-8") as f:
                schema_info.append(f.read())
        else:
            raise FileNotFoundError(f"DDL.csv not found in {base}")
    
    if not schema_info:
        raise ValueError(f"No valid schema content found for {db_id}")
    
    # 合并所有schema信息
    combined_schema = "\n\n" + "="*80 + "\n\n".join(schema_info)
    
    print(f"✅ Loaded schema for {db_id} with {len(schema_info)} sections")
    return combined_schema



def extract_sqlite_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema = {}

    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        # 获取列名、数据类型、是否主键
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()

        # 获取外键
        cursor.execute(f"PRAGMA foreign_key_list('{table}')")
        fks = cursor.fetchall()

        schema[table] = {
            "columns": [
                {
                    "name": col[1],
                    "type": col[2],
                    "is_pk": bool(col[5])
                }
                for col in columns
            ],
            "foreign_keys": [
                {
                    "from": fk[3],
                    "to_table": fk[2],
                    "to_column": fk[4]
                }
                for fk in fks
            ]
        }

    conn.close()
    return schema

def format_schema_readable(schema):
    lines = []
    for table, info in schema.items():
        lines.append(f"Table: {table}")
        for col in info["columns"]:
            pk = " [PK]" if col["is_pk"] else ""
            lines.append(f"- {col['name']}{pk}")
        lines.append("")  # blank line for spacing

    lines.append("Foreign Keys:")
    for table, info in schema.items():
        for fk in info["foreign_keys"]:
            lines.append(f"- {table}.{fk['from']} → {fk['to_table']}.{fk['to_column']}")

    return "\n".join(lines)

def sqlite_schema_extract_format(db_path):
    """Extract and format SQLite schema for readability, can directly use in prompts"""
    schema = extract_sqlite_schema(db_path)
    return format_schema_readable(schema)




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