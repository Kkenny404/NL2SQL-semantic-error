import sqlite3

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





if __name__ == "__main__":
    db_path = "BIRD/dev_20240627/dev_databases/financial/financial.sqlite"
    formatted = sqlite_schema_extract_format(db_path)
    print(formatted)

    # 写入文件
    with open("financial_schema.txt", "w", encoding="utf-8") as f:
        f.write(formatted)

    print("✅ Schema written to financial_schema.txt")
