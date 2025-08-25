import json
import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison, Parenthesis, Function
from sqlparse.tokens import Keyword, DML

# === 提取列名/表名 ===
def extract_identifiers_compact(token_list):
    identifiers = []
    if isinstance(token_list, IdentifierList):
        for identifier in token_list.get_identifiers():
            if isinstance(identifier, Identifier):
                identifiers.append(identifier.get_real_name() or str(identifier))
            else:
                identifiers.append(str(identifier))
    elif isinstance(token_list, Identifier):
        identifiers.append(token_list.get_real_name() or str(token_list))
    elif isinstance(token_list, Function):
        identifiers.append(str(token_list.get_name()))
    else:
        identifiers.append(str(token_list))
    return identifiers

# === 解析 WHERE ===
def parse_where_compact(where_token):
    conditions = []
    for token in where_token.tokens:
        if isinstance(token, Comparison):
            left_token = token.left
            if isinstance(left_token, Identifier):
                col = left_token.get_real_name()
            else:
                col = str(left_token).strip()
            conditions.append({
                "column": col,
                "type": "comparison"
            })
        elif isinstance(token, Parenthesis):
            conditions.append({"type": "nested_condition"})
    return conditions

# === 解析 CTE（WITH 子句） ===
def parse_cte(sql_text, max_depth=2, current_depth=0):
    """
    提取 WITH 子句，返回 [{'name': cte_name, 'query': parsed_sql}, ...]
    """
    ctes = []
    # 匹配 WITH cte_name AS ( ... )，支持多个 CTE
    pattern = r"WITH\s+([\w]+)\s+AS\s*\((.*?)\)(?=(\s*,\s*[\w]+\s+AS\s*\()|\s*SELECT|\Z)"
    matches = re.finditer(pattern, sql_text, re.IGNORECASE | re.DOTALL)
    for match in matches:
        cte_name, cte_sql = match.group(1), match.group(2)
        ctes.append({
            "name": cte_name,
            "query": parse_sql_compact(cte_sql.strip(), max_depth, current_depth + 1)
        })
    return ctes

# === 递归解析 SQL ===
def parse_sql_compact(sql, max_depth=2, current_depth=0):
    if current_depth > max_depth:
        return {"type": f"NestedSubquery(level={current_depth})"}

    try:
        parsed = sqlparse.parse(sql)[0]
    except Exception as e:
        return {"parse_error": str(e)}

    result = {
        "select": [],
        "from": [],
        "where": [],
        "group_by": [],
        "subqueries": [],
        "ctes": []
    }

    # 先提取 CTE
    result["ctes"] = parse_cte(sql, max_depth, current_depth)

    tokens = [t for t in parsed.tokens if not t.is_whitespace]
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]

        if token.ttype is DML and token.value.upper() == "SELECT":
            idx += 1
            select_fields = extract_identifiers_compact(tokens[idx])
            result["select"].extend(select_fields)

        elif token.ttype is Keyword and token.value.upper() == "FROM":
            idx += 1
            result["from"].extend(extract_identifiers_compact(tokens[idx]))

        elif isinstance(token, Where):
            result["where"].extend(parse_where_compact(token))

        elif token.ttype is Keyword and token.value.upper() == "GROUP BY":
            idx += 1
            result["group_by"].extend(extract_identifiers_compact(tokens[idx]))

        elif isinstance(token, Parenthesis):
            subquery = str(token).strip("() ")
            if "SELECT" in subquery.upper():
                result["subqueries"].append(
                    parse_sql_compact(subquery, max_depth=max_depth, current_depth=current_depth + 1)
                )

        idx += 1

    return result

# === 批量处理文件 ===
input_file = "bug-data/NL2SQL-Bugs.json"
output_file = "bug-data/NL2SQL-Bugs-parsed.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

parsed_data = []
for item in data:
    parsed_entry = {
        "id": item.get("id"),
        "original_sql": item.get("sql", ""),
        "parsed_sql": parse_sql_compact(item.get("sql", ""), max_depth=2)
    }
    parsed_data.append(parsed_entry)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(parsed_data, f, ensure_ascii=False, indent=2)

print(f"Compact parsed file with CTE support saved to {output_file}")
