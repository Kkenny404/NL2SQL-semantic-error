# # utils.py
import sqlite3
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic
from openai import OpenAI
from schema.get_spider_schema import get_schema, schema_shrink
import sqlglot
from sqlglot import parse_one
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


# Anthropic 配置
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
claude_client = anthropic.Anthropic(api_key=claude_api_key)

# OpenAI 配置
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

gen_id = "gemini-2.5-flash"  # 使用 Gemini 2.5 Flash 模型   
GEN_model = genai.GenerativeModel(gen_id)

def get_db_ids_from_json():
    """从 NL2SQL-Bugs.json 中提取所有 db_id"""
    with open("data/NL2SQL-Bugs.json", "r") as f:
        data = json.load(f)

    db_ids = sorted(set(example["db_id"] for example in data))
    print(f"共需要 {len(db_ids)} 个数据库：")
    for db in db_ids:
        print(db)

# NOTE：提取 SQLite 数据库模式，19/07之间都是使用这个版本，但是并不能完全满足需求，只能提取表名和列名，没有外键等信息
# 后面更新一个schema_extract_test.py来提取更详细的模式信息
def extract_schema_from_sqlite(sqlite_path: str) -> str:
    """Extract table and column info from .sqlite database"""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']
    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        cols = [col[1] for col in  cursor.fetchall()]
        schema_parts.append(f"{table}({', '.join(cols)})")
    conn.close()
    return " | ".join(schema_parts)


def build_prompt(question: str, schema: str, sql: str) -> str:
    """Fill in the prompt template"""
    return f"""You are a database expert.

Given the following natural language question, SQL query, and database schema, please determine whether the SQL is semantically correct with respect to the question.

You need to answer with "Yes" or "No" only, without outputting any explanation or additional text.

Question: {question}

Schema: {schema}

SQL: {sql}

Is the SQL semantically correct?"""


def build_prompt_no_schema(question: str, schema: str, sql: str) -> str:
    return f"""You are a database expert.

Given the following natural language question and the SQL query. Please determine whether the SQL is semantically correct with respect to the question.

You need to answer with "Yes" or "No" only, without outputting any explanation or additional text.

Question: {question}

SQL: {sql}

Is the SQL semantically correct?"""


def build_cot_prompt(question: str, schema: str, sql: str) -> str:
    return f"""You are a database expert. Analyze whether the SQL query correctly answers the natural language question. 

Question: {question}

Database Schema: {schema}

SQL Query: {sql}

Analyze this step by step in your mind (do not output the steps, just the final answer):

1. Question Understanding: What exactly is the question asking for?
   - What information should be retrieved?
   - Are there any specific conditions or constraints?

2. SQL Logic Analysis: Break down what the SQL actually does:
   - Which tables and columns are selected?
   - What are the JOIN conditions?
   - What filters are applied in WHERE clause?
   - Are aggregations/sorting correctly used?

3. Semantic Correctness Check: 
   - Do the selected columns semantically match what the question requests?
   - Are the table relationships and JOIN logic semantically appropriate?
   - Do the WHERE conditions semantically align with the question's intent?
   - Does the overall query logic semantically represent the question's meaning?

4. Final Verification: Does the SQL query produce the exact information requested in the question?

Final Answer: [Yes or No]. Answer one word only, without any explanation or additional text.

"""

def query_gemini(prompt: str) -> str:
    print(f"[Gemini] Querying model: {gen_id}")
    response = GEN_model.generate_content(prompt)
    return response.text.strip().lower()


def parse_answer(text: str) -> bool:
    """Parse model's Yes/No answer to boolean - improved version"""
    text = text.lower().strip()
    
    # 按行分割，从后往前找第一个包含 yes/no 的行
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if 'yes' in line and 'no' not in line:
            return True
        elif 'no' in line and 'yes' not in line:
            return False
    
    # 如果按行找不到，用原来的逻辑
    if "yes" in text and "no" not in text:
        return True
    elif "no" in text and "yes" not in text:
        return False
    else:
        return None
    

def query_claude(prompt: str) -> str:
    """Call Claude 4 sonnet"""
    print("[Claude] Querying model: claude-sonnet-4-20250514")
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        temperature=0,  # Lower temperature for deterministic output
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text.strip().lower()

def query_gpt(prompt: str, model: str = "gpt-4.1") -> str:
    """Call GPT model using Chat Completions API"""
    print(f"[GPT] Querying model: {model}")
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0,  # 确定性输出
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"[ERROR] GPT API call failed: {e}")
        raise e



def load_sql(sql_root, sql_id):
    """Load the SQL content from a file based on the given SQL ID."""
    sql_path = os.path.join(sql_root, f"{sql_id}")
    if not os.path.exists(sql_path):
        print(f"[SKIP] SQL file not found: {sql_path}")
        return None
    with open(sql_path, "r", encoding="utf-8") as f:
        return f.read()



def parser_sql(origin_sql: str) -> str:
    """Parse SQL into a simplified AST (meta removed) and return as JSON string"""
    ast = parse_one(origin_sql)  # 修正变量名

    # 转 dict
    ast_dict = ast.to_dict()

    # 去掉 meta
    def strip_meta(node):
        if isinstance(node, dict):
            return {k: strip_meta(v) for k, v in node.items() if k != "meta"}
        elif isinstance(node, list):
            return [strip_meta(i) for i in node]
        else:
            return node

    clean_ast = strip_meta(ast_dict)
    return json.dumps(clean_ast, indent=2)  # 返回 JSON 字符串

def parser_call_prompt(question: str, schema: str, sql: str) -> str:
    """Build a prompt for the SQL parser"""
    new_sql_ast = parser_sql(sql)  # Parse SQL to JSON format
    return build_prompt(question, schema, new_sql_ast)
   





if __name__ == "__main__":
    # 简单测试
    simple_prompt = "Answer with 'Yes' or 'No' only. Is 2+2=4?"
    result = query_gemini(simple_prompt)
    print(f"[GPT Simple Test]: '{result}'")


#     # SQL 测试
#     sql_prompt = """You are a database expert.
# Question: What is the name of the student with ID 1?
# Schema: students(id, name, age)
# SQL: SELECT name FROM students WHERE id = 1;
# Answer 'Yes' or 'No' only: Is this SQL correct?"""
    
#     result2 = query_gemini(sql_prompt)
#     print(f"[Gemini SQL Test]: '{result2}'")



