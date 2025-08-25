import sqlglot
from sqlglot import parse_one
from sqlglot.expressions import Subquery, Table, Select, Where, Group, Order

def detect_dialect(sql_id: str) -> str:
    """根据 sql_id/db_id 前缀猜方言"""
    sql_id_lower = sql_id.lower()
    if sql_id_lower.startswith("bq") or "ga" in sql_id_lower:
        return "bigquery"
    elif sql_id_lower.startswith("sf") or "snowflake" in sql_id_lower:
        return "snowflake"
    elif sql_id_lower.startswith("local") or "sqlite" in sql_id_lower:
        return "sqlite"
    return None  # 默认宽松模式

def parse_sql_glot_multi(sql: str, sql_id: str, max_depth=2, current_depth=0):
    """多方言 SQL 解析 + 容错 + 递归"""
    if current_depth > max_depth:
        return {"type": f"Nested(level={current_depth})"}

    dialect = detect_dialect(sql_id)
    tree = None

    # 多方言 + 回退
    for d in [dialect, "duckdb", None]:
        try:
            tree = parse_one(sql, read=d, error_level="ignore")
            if tree:
                break
        except:
            continue

    if tree is None:
        return {
            "parse_error": "Could not parse SQL",
            "raw_sql": sql
        }

    result = {
        "select": [str(s) for s in tree.find_all(Select)],
        "from": [str(t) for t in tree.find_all(Table)],
        "where": [str(w.this) for w in tree.find_all(Where)],
        "group_by": [str(g) for g in tree.find_all(Group)],
        "order_by": [str(o) for o in tree.find_all(Order)],
        "ctes": [],
        "subqueries": []
    }

    # 递归 CTE
    if tree.args.get("with"):
        for cte in tree.args["with"].expressions:
            cte_name = cte.alias
            cte_sql = cte.this.sql()
            result["ctes"].append({
                "name": cte_name,
                "query": parse_sql_glot_multi(cte_sql, sql_id, max_depth=max_depth, current_depth=current_depth + 1)
            })

    # 递归 Subquery
    for subq in tree.find_all(Subquery):
        subq_sql = subq.this.sql()
        result["subqueries"].append(
            parse_sql_glot_multi(subq_sql, sql_id, max_depth=max_depth, current_depth=current_depth + 1)
        )

    return result
