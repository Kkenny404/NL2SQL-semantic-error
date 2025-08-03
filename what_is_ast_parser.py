# import sqlglot
# from sqlglot import parse_one

# # 示例 SQL
# sql = "SELECT u.id, u.name, COUNT(o.id) AS order_count FROM users u JOIN orders o ON u.id = o.user_id WHERE u.status = 'active' GROUP BY u.id, u.name"

# # 解析 SQL
# ast = parse_one(sql)

# # 打印 AST 简单形式
# print("=== AST (Pretty) ===")
# print(ast)

# # 打印更详细的树状结构
# print("\n=== AST (Tree View) ===")
# print(ast.dump())
