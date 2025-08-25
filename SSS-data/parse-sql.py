import os
import json
from tqdm import tqdm
from parsed_sql_with_CTE import parse_sql_glot_multi

SQL_ROOT = "SSS-data/sql"
DATA_PATH = "SSS-data/random_order_ground_truth.json"
OUTPUT_PATH = "SSS-data/parsed_sql.json"

with open(DATA_PATH, "r") as f:
    examples = json.load(f)

parsed_results = []
for ex in tqdm(examples):
    sql_id = ex["sql_id"]
    sql_path = os.path.join(SQL_ROOT, sql_id)
    if not os.path.exists(sql_path):
        print(f"[WARN] SQL file not found: {sql_path}")
        continue
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    parsed_sql = parse_sql_glot_multi(sql, sql_id, max_depth=2)
    parsed_results.append({
        "sql_id": sql_id,
        "parsed_sql": parsed_sql
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(parsed_results, f, ensure_ascii=False, indent=2)

print(f"Parsed SQL file saved to {OUTPUT_PATH}")
