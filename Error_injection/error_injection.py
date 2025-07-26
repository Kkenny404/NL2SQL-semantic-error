import json
import random
import os
import re
from pathlib import Path
import sys
import glob
from typing import Callable, List, Dict, Any
from abc import ABC, abstractmethod
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Error_injection.filters import ErrorCategoryManager
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
        sqlite_path = os.path.join(schema_root, "spider2-localdb", f"{db_id}.sqlite")
        print(f"Trying to open SQLite file at: {sqlite_path}")
        return sqlite_schema_extract_format(sqlite_path)
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


def get_schema_summary(instance_id, db_id, schema_root):
    """
    获取schema的摘要信息, 用于快速了解数据库结构
    """
    try:
        full_schema = get_schema(instance_id, db_id, schema_root, include_all_related=True)
        
        # 统计表的数量
        table_count = len(re.findall(r"CREATE TABLE", full_schema, re.IGNORECASE))
        
        # 提取所有表名
        table_names = re.findall(r"CREATE TABLE [`\"]?([^`\"\s\(]+)", full_schema, re.IGNORECASE)
        
        # 按数据库分组表名（如果有路径信息）
        schema_sections = full_schema.split("-- Schema from:")
        section_info = []
        
        for section in schema_sections[1:]:  # 跳过第一个空section
            lines = section.strip().split('\n')
            if lines:
                path_info = lines[0].strip()
                section_tables = re.findall(r"CREATE TABLE [`\"]?([^`\"\s\(]+)", section, re.IGNORECASE)
                section_info.append({
                    "path": path_info,
                    "table_count": len(section_tables),
                    "tables": section_tables[:5]  # 只显示前5个表名
                })
        
        summary = f"""
Database: {db_id}
Total Tables: {table_count}
Schema Sections: {len(section_info)}

Section Details:
"""
        for info in section_info:
            summary += f"  📁 {info['path']}: {info['table_count']} tables"
            if info['tables']:
                summary += f" (e.g., {', '.join(info['tables'])})"
            summary += "\n"
        
        return summary
        
    except Exception as e:
        return f"❌ Error getting schema summary: {e}"

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



# ========== Prompt 生成器 ==========

def build_prompt(question: str,
        correct_sql: str,
        schema: str,
        category: str,
        category_description: str,
        sub_error_type: str,
        error_description: str) -> str:
    """构造用于生成包含特定语义错误的 SQL 的 prompt"""
    
    prompt = f"""# SQL Semantic Error Injection Task

You are an expert SQL developer tasked with injecting a SPECIFIC semantic error into a correct SQL query.

## 🎯 Your Mission
Take the provided correct SQL query and modify it to contain EXACTLY ONE semantic error of the specified type. The modified SQL should:
- ✅ Have valid syntax (no syntax errors)
- ✅ Be executable against the database
- ❌ Contain the specified semantic error
- ❌ NOT contain any other types of errors

## 📋 Error Specification
**Error Category:** {category}
**Category Description:** {category_description}

**Specific Error Type:** {sub_error_type}
**Error Description:** {error_description}

## 💡 Guidelines for Error Injection

### DO:
- Focus ONLY on the specified error type
- Keep the query structure as similar as possible to the original
- Ensure the modified SQL runs without syntax errors
- Make the error subtle but semantically meaningful
- Consider the business logic and question intent when injecting the error

### DON'T:
- Add syntax errors or typos
- Introduce multiple different error types
- Completely rewrite the query structure
- Make obvious/trivial mistakes that wouldn't occur in real scenarios
- Change unrelated parts of the query

## 📊 Database Schema
```sql
{schema}
```

## ❓ Natural Language Question
"{question}"

## ✅ Correct SQL Query
```sql
{correct_sql}
```

## 🔧 Your Task
Analyze the correct SQL query above and modify it to introduce the specific error type: **{sub_error_type}**.

Think about:
1. How this error typically manifests in real-world scenarios
2. What parts of the query are most relevant to this error type
3. How to make this data be meaningful for improving LLM SQL semantic detection area

## 📤 Output Format
Return your response in the following format:

**Modified SQL Query:**
```sql
[Your modified SQL query here]
```

**Brief Explanation:**
[A very brief explanation (1-2 sentences) of what specific change you made to inject the error]

"""
    
    return prompt

# ========== 配置类 ==========

class SchemaConfig:
    """Schema处理配置"""
    
    def __init__(self, 
                 include_all_related: bool = True,
                 shrink_schema: bool = False,
                 auto_shrink_threshold_lines: int = 700,  # 自动shrink的行数阈值
                 max_schema_size: int = 200000,  # 最大schema字符数
                 show_schema_summary: bool = True):
        self.include_all_related = include_all_related
        self.shrink_schema = shrink_schema
        self.auto_shrink_threshold_lines = auto_shrink_threshold_lines
        self.max_schema_size = max_schema_size
        self.show_schema_summary = show_schema_summary
    
    def process_schema(self, schema: str, instance_id: str) -> str:
        """处理schema"""
        processed_schema = schema
        
        # 统计schema行数
        schema_lines = len(schema.strip().split('\n'))
        
        # 检查是否需要shrink
        should_shrink = self.shrink_schema
        shrink_reason = "manually enabled"
        
        # 自动shrink逻辑：只有在超过阈值时才shrink
        if not should_shrink and schema_lines > self.auto_shrink_threshold_lines:
            should_shrink = True
            shrink_reason = f"auto (schema has {schema_lines} lines > {self.auto_shrink_threshold_lines} threshold)"
        
        # 应用shrink
        if should_shrink:
            print(f"    🔄 Applying schema shrink for {instance_id} - {shrink_reason}")
            original_lines = schema_lines
            processed_schema = schema_shrink(processed_schema)
            new_lines = len(processed_schema.strip().split('\n'))
            print(f"    📊 Schema reduced from {original_lines} to {new_lines} lines")
        else:
            print(f"    📋 Keeping full schema for {instance_id} ({schema_lines} lines)")
        
        # 检查最终schema大小
        if len(processed_schema) > self.max_schema_size:
            print(f"    ⚠️ Warning: Schema for {instance_id} is still large ({len(processed_schema)} chars)")
            print(f"       Consider increasing auto_shrink_threshold_lines or max_schema_size")
        
        return processed_schema
    
    @classmethod
    def create_conservative_config(cls, line_threshold: int = 700):
        """创建保守配置：保持完整schema，只在超过阈值时shrink"""
        return cls(
            include_all_related=True,
            shrink_schema=False,
            auto_shrink_threshold_lines=line_threshold,
            max_schema_size=300000,  # 更大的字符限制
            show_schema_summary=True
        )
    
    @classmethod  
    def create_aggressive_config(cls, line_threshold: int = 300):
        """创建激进配置：更容易触发shrink"""
        return cls(
            include_all_related=True,
            shrink_schema=False,
            auto_shrink_threshold_lines=line_threshold,
            max_schema_size=100000,
            show_schema_summary=True
        )
    
    @classmethod
    def create_always_shrink_config(cls):
        """创建总是shrink的配置"""
        return cls(
            include_all_related=True,
            shrink_schema=True,  # 强制shrink
            auto_shrink_threshold_lines=999999,  # 高阈值避免重复shrink
            max_schema_size=500000,
            show_schema_summary=True
        )


# ========== 主要处理类 ==========

class ErrorInjectionProcessor:
    """错误注入处理器"""
    
    def __init__(self, 
                 error_explanation_path: str,
                 schema_root: str,
                 sql_root: str,
                 output_dir: str,
                 schema_config: SchemaConfig = None):
        self.error_explanation_path = error_explanation_path
        self.schema_root = schema_root
        self.sql_root = sql_root
        self.output_dir = output_dir
        self.category_manager = ErrorCategoryManager()
        self.schema_config = schema_config or SchemaConfig()
        
    def generate_prompts_for_category(self,
                                    category_name: str,
                                    jsonl_path: str,
                                    sample_size: int = 10):
        """为指定类别生成 prompts(不调用LLM)"""
        
        print(f"🔄 Processing category: {category_name}")
        
        # 创建输出目录
        category_output_dir = os.path.join(self.output_dir, category_name)
        os.makedirs(category_output_dir, exist_ok=True)
        
        # 加载数据
        error_data = load_error_explanation(self.error_explanation_path)
        dataset = load_jsonl(jsonl_path)
        sub_errors = get_sub_error_types(error_data, category_name)
        
        if not sub_errors:
            print(f"❌ No sub-errors found for category: {category_name}")
            return
        
        # 获取类别描述
        category_description = ""
        for cat in error_data:
            if cat["category"] == category_name:
                category_description = cat["description"]
                break
        
        ground_truth_records = []
        
        for sub_error in sub_errors:
            sub_error_type = sub_error["error_type"]
            sub_error_description = sub_error["description"]
            
            print(f"  📝 Processing sub-error: {sub_error_type}")
            
            # 获取对应的过滤策略
            filter_strategy = self.category_manager.get_filter_strategy(category_name, sub_error_type)
            filtered_dataset = filter_strategy.filter(dataset, self.sql_root)
            
            if len(filtered_dataset) == 0:
                print(f"    ⚠️ No data after filtering for {sub_error_type}")
                continue
            
            # 采样数据
            actual_sample_size = min(sample_size, len(filtered_dataset))
            sampled_data = random.sample(filtered_dataset, k=actual_sample_size)
            
            for i, item in enumerate(sampled_data):
                try:
                    instance_id = item["instance_id"]
                    db = item["db"]
                    question = item["question"]
                    external_knowledge = item.get("external_knowledge", "")
                    
                    # 加载SQL
                    true_sql_path = os.path.join(self.sql_root, f"{instance_id}.sql")
                    if not os.path.exists(true_sql_path):
                        print(f"    ⚠️ SQL file not found: {true_sql_path}")
                        continue
                    
                    true_sql = read_file(true_sql_path)
                    
                    # 加载Schema
                    try:
                        # 获取完整的schema（包含所有相关子数据库）
                        schema = get_schema(
                            instance_id, 
                            db, 
                            self.schema_root, 
                            include_all_related=self.schema_config.include_all_related
                        )
                        
                        # 打印schema摘要信息
                        if i == 0 and self.schema_config.show_schema_summary:
                            schema_summary = get_schema_summary(instance_id, db, self.schema_root)
                            print(f"    📊 Schema summary for {db}:\n{schema_summary}")
                        
                        # 处理schema（简化、大小检查等）
                        schema = self.schema_config.process_schema(schema, instance_id)
                        
                    except Exception as e:
                        print(f"    ⚠️ Failed to load schema for {instance_id}: {e}")
                        continue
                    
                    # 生成prompt
                    try:
                        prompt = build_prompt(
                            question=question,
                            correct_sql=true_sql,
                            schema=schema,
                            category=category_name,
                            category_description=category_description,
                            sub_error_type=sub_error_type,
                            error_description=sub_error_description
                        )
                        
                        if prompt is None:
                            print(f"    ⚠️ Failed to generate prompt for {instance_id}")
                            continue
                            
                    except Exception as e:
                        print(f"    ⚠️ Error generating prompt for {instance_id}: {e}")
                        continue
                    
                    # 保存prompt
                    prompt_filename = f"{instance_id}_{sub_error_type.replace(' ', '')}_{i+1}_prompt.txt"
                    prompt_path = os.path.join(category_output_dir, prompt_filename)
                    
                    try:
                        with open(prompt_path, "w", encoding="utf-8") as f:
                            f.write(prompt)
                        print(f"    ✅ Generated prompt: {prompt_filename}")
                    except Exception as e:
                        print(f"    ⚠️ Error saving prompt for {instance_id}: {e}")
                        continue
                    
                    
                    # 记录ground truth信息
                    sql_filename = prompt_filename.replace("_prompt.txt", ".sql")
                    ground_truth_records.append(
                        format_ground_truth_record(
                            true_sql_id=instance_id,
                            sql_filename=sql_filename,
                            db=db,
                            question=question,
                            category_name=category_name,
                            sub_error_type=sub_error_type,
                            external_knowledge=external_knowledge
                        )
                    )
                    
                except Exception as e:
                    print(f"    ⚠️ Error processing {item['instance_id']}: {e}")
        
        # 保存ground truth
        ground_truth_path = os.path.join(category_output_dir, "ground_truth.jsonl")
        with open(ground_truth_path, "w", encoding="utf-8") as f:
            for record in ground_truth_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"✅ Category {category_name} processed. Generated {len(ground_truth_records)} prompts.")
    
    def list_available_categories(self) -> List[str]:
        """列出所有可用的错误类别"""
        return self.category_manager.get_available_categories()


def format_ground_truth_record(true_sql_id: str, 
                              sql_filename: str,
                              db: str, 
                              question: str,
                              category_name: str,
                              sub_error_type: str,
                              external_knowledge: str = None) -> Dict:
    """格式化ground truth记录，确保格式一致"""
    return {
        "true_sql_id": true_sql_id,
        "sql_id": sql_filename,
        "db": db,
        "question": question,
        "error_types": [
            {
                "error_type": category_name,
                "sub_error_type": sub_error_type
            }
        ],
        "external_knowledge": external_knowledge
    }



# ========== LLM响应解析器 ==========

def parse_llm_response(response: str) -> tuple[str, str]:
    """
    解析LLM的响应，提取SQL和解释
    
    Args:
        response: LLM的原始响应
    
    Returns:
        tuple: (sql_query, explanation)
    """
    import re
    
    # 尝试提取SQL（在```sql和```之间）
    sql_pattern = r'```sql\s*(.*?)\s*```'
    sql_matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if sql_matches:
        sql_query = sql_matches[0].strip()
    else:
        # 如果没有找到SQL代码块，尝试其他模式
        sql_pattern_alt = r'\*\*Modified SQL Query:\*\*\s*(.*?)(?=\*\*Brief Explanation:\*\*|$)'
        sql_matches_alt = re.findall(sql_pattern_alt, response, re.DOTALL | re.IGNORECASE)
        if sql_matches_alt:
            sql_query = sql_matches_alt[0].strip()
        else:
            # 最后的备选方案：取整个响应的前半部分
            lines = response.strip().split('\n')
            sql_lines = []
            for line in lines:
                if '**Brief Explanation:**' in line or 'explanation:' in line.lower():
                    break
                sql_lines.append(line)
            sql_query = '\n'.join(sql_lines).strip()
    
    # 提取解释
    explanation_pattern = r'\*\*Brief Explanation:\*\*\s*(.*?)(?=\n\s*$|$)'
    explanation_matches = re.findall(explanation_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if explanation_matches:
        explanation = explanation_matches[0].strip()
    else:
        # 备选方案：查找包含"explanation"的行
        lines = response.strip().split('\n')
        explanation_lines = []
        found_explanation = False
        for line in lines:
            if 'explanation:' in line.lower() or found_explanation:
                found_explanation = True
                explanation_lines.append(line)
        explanation = '\n'.join(explanation_lines).strip()
        if not explanation:
            explanation = "No explanation provided"
    
    # 清理SQL
    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
    
    # 清理解释
    explanation = explanation.replace('**Brief Explanation:**', '').strip()
    
    return sql_query, explanation


# ========== LLM调用器（分离的模块）==========

class LLMErrorInjector:
    """LLM错误注入器 - 独立的模块"""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 你的LLM客户端, 比如OpenAI客户端
        """
        self.llm_client = llm_client
    
    def inject_error_with_llm(self, prompt: str) -> tuple[str, str]:
        """
        使用LLM注入错误
        
        Returns:
            tuple: (error_sql, explanation)
        """
        if self.llm_client is None:
            raise ValueError("LLM client not provided")
        
        try:
            # OpenAI API调用
            response = self.llm_client.chat.completions.create(
                model="gpt-4.1",  # 或者 "gpt-3.5-turbo"
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert SQL developer who specializes in injecting semantic errors into SQL queries for training purposes."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # 低温度保证一致性
                max_tokens=1000   # 根据需要调整
            )
            
            llm_response = response.choices[0].message.content
            
            # 解析LLM响应
            error_sql, explanation = parse_llm_response(llm_response)
            return error_sql, explanation
            
        except Exception as e:
            print(f"❌ LLM API error: {e}")
            # 返回错误信息
            return f"-- Error calling LLM: {e}", f"LLM call failed: {e}"
    
    def process_prompts_in_directory(self, 
                                   category_dir: str,
                                   output_sql_dir: str = None,
                                   output_explanation_dir: str = None):
        """
        处理目录中的所有prompt文件，生成SQL和解释文件
        
        Args:
            category_dir: 包含prompt文件的目录
            output_sql_dir: SQL文件输出目录
            output_explanation_dir: 解释文件输出目录
        """
        if output_sql_dir is None:
            output_sql_dir = os.path.join(category_dir, "error_sqls")
        if output_explanation_dir is None:
            output_explanation_dir = os.path.join(category_dir, "explanations")
        
        os.makedirs(output_sql_dir, exist_ok=True)
        os.makedirs(output_explanation_dir, exist_ok=True)
        
        # 查找所有prompt文件
        prompt_files = glob.glob(os.path.join(category_dir, "*_prompt.txt"))
        
        generated_records = []
        
        for prompt_file in prompt_files:
            try:
                prompt = read_file(prompt_file)
                error_sql, explanation = self.inject_error_with_llm(prompt)
                
                # 生成对应的文件名
                base_name = os.path.basename(prompt_file).replace("_prompt.txt", "")
                
                # 保存SQL文件
                sql_filename = f"{base_name}.sql"
                sql_path = os.path.join(output_sql_dir, sql_filename)
                with open(sql_path, "w", encoding="utf-8") as f:
                    f.write(error_sql.strip())
                
                # 保存解释文件
                explanation_filename = f"{base_name}_explanation.txt"
                explanation_path = os.path.join(output_explanation_dir, explanation_filename)
                with open(explanation_path, "w", encoding="utf-8") as f:
                    f.write(explanation.strip())
                
                # 记录生成的文件
                generated_records.append({
                    "prompt_file": prompt_file,
                    "sql_file": sql_path,
                    "explanation_file": explanation_path,
                    "base_name": base_name
                })
                
                print(f"✅ Generated error SQL: {sql_path}")
                print(f"✅ Generated explanation: {explanation_path}")
                
            except Exception as e:
                print(f"⚠️ Error processing {prompt_file}: {e}")
        
        # 保存生成记录
        records_path = os.path.join(category_dir, "generated_files_record.jsonl")
        with open(records_path, "w", encoding="utf-8") as f:
            for record in generated_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"📝 Generated {len(generated_records)} SQL-explanation pairs")
        print(f"📋 Records saved to: {records_path}")
        
    def process_category_with_llm(self, 
                                category_dir: str,
                                organize_by_error_type: bool = True):
        """
        处理整个错误类别目录，可选择按错误类型组织文件
        
        Args:
            category_dir: 错误类别目录
            organize_by_error_type: 是否按错误类型创建子目录
        """
        if organize_by_error_type:
            # 按错误类型组织文件结构
            self._process_with_organization(category_dir)
        else:
            # 简单处理，所有文件放在一起
            self.process_prompts_in_directory(category_dir)
    
    def _process_with_organization(self, category_dir: str):
        """按错误类型组织文件的处理方法"""
        prompt_files = glob.glob(os.path.join(category_dir, "*_prompt.txt"))
        
        if not prompt_files:
            print(f"❌ No prompt files found in {category_dir}")
            return
        
        # 按错误类型分组文件
        error_type_groups = {}
        for prompt_file in prompt_files:
            filename = os.path.basename(prompt_file)
            # 从文件名提取错误类型（假设格式为：instanceid_ErrorType_num_prompt.txt）
            parts = filename.replace("_prompt.txt", "").split("_")
            if len(parts) >= 2:
                error_type = parts[1]  # 获取错误类型部分
                if error_type not in error_type_groups:
                    error_type_groups[error_type] = []
                error_type_groups[error_type].append(prompt_file)
        
        all_generated_records = []
        
        for error_type, files in error_type_groups.items():
            print(f"\n🔄 Processing error type: {error_type}")
            
            # 为每个错误类型创建子目录
            error_type_dir = os.path.join(category_dir, error_type)
            sql_dir = os.path.join(error_type_dir, "error_sqls")
            explanation_dir = os.path.join(error_type_dir, "explanations")
            
            os.makedirs(sql_dir, exist_ok=True)
            os.makedirs(explanation_dir, exist_ok=True)
            
            for prompt_file in files:
                try:
                    prompt = read_file(prompt_file)
                    error_sql, explanation = self.inject_error_with_llm(prompt)
                    
                    # 生成文件名
                    base_name = os.path.basename(prompt_file).replace("_prompt.txt", "")
                    
                    # 保存SQL文件
                    sql_filename = f"{base_name}.sql"
                    sql_path = os.path.join(sql_dir, sql_filename)
                    with open(sql_path, "w", encoding="utf-8") as f:
                        f.write(error_sql.strip())
                    
                    # 保存解释文件
                    explanation_filename = f"{base_name}_explanation.txt"
                    explanation_path = os.path.join(explanation_dir, explanation_filename)
                    with open(explanation_path, "w", encoding="utf-8") as f:
                        f.write(explanation.strip())
                    
                    all_generated_records.append({
                        "error_type": error_type,
                        "prompt_file": prompt_file,
                        "sql_file": sql_path,
                        "explanation_file": explanation_path,
                        "base_name": base_name
                    })
                    
                    print(f"  ✅ {base_name}: SQL + explanation generated")
                    
                except Exception as e:
                    print(f"  ⚠️ Error processing {prompt_file}: {e}")
        
        # 保存整体记录
        master_records_path = os.path.join(category_dir, "master_generated_files_record.jsonl")
        with open(master_records_path, "w", encoding="utf-8") as f:
            for record in all_generated_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"\n🎯 Summary:")
        print(f"📊 Total files generated: {len(all_generated_records)}")
        print(f"📋 Error types processed: {list(error_type_groups.keys())}")
        print(f"📝 Master record saved to: {master_records_path}")
        
        return all_generated_records


# ========== 主函数示例 ==========

def main():
    """主函数示例"""
    
    # 配置参数
    error_explanation_path = "Spider/error_explaination.json"
    jsonl_path = "Spider/spider2-lite_with_golden.jsonl"
    schema_root = "spider2-lite/resource/databases"
    sql_root = "Spider/lite_256_true_sql"
    output_dir = "Error_injection/Error_data"
    
    # 配置schema处理选项 - 三种方式任选其一
    
    # 方式1: 手动配置 (推荐给你的场景)
    schema_config = SchemaConfig(
        include_all_related=True,           # 包含所有相关的子数据库schema
        shrink_schema=False,                # 不强制简化schema
        auto_shrink_threshold_lines=700,    # 只有超过700行时才自动shrink
        max_schema_size=200000,             # 允许更大的schema (字符数)
        show_schema_summary=True            # 显示schema摘要
    )
    
    # 方式2: 使用预设配置
    # schema_config = SchemaConfig.create_conservative_config(line_threshold=700)
    
    # 方式3: 如果你想更激进地控制schema大小
    # schema_config = SchemaConfig.create_aggressive_config(line_threshold=500)
    
    # 创建处理器
    processor = ErrorInjectionProcessor(
        error_explanation_path=error_explanation_path,
        schema_root=schema_root,
        sql_root=sql_root,
        output_dir=output_dir,
        schema_config=schema_config
    )
    
    # 列出可用类别
    available_categories = processor.list_available_categories()
    print("Available categories:")
    for i, category in enumerate(available_categories, 1):
        print(f"  {i}. {category}")
    
    # 选择要处理的类别（可以通过用户输入控制）
    target_category = "Table-related Errors"  # 可以改为用户输入
    
    if target_category in available_categories:
        # 生成prompts
        processor.generate_prompts_for_category(
            category_name=target_category,
            jsonl_path=jsonl_path,
            sample_size=10
        )
        
        # 可选：使用LLM生成错误SQL和解释
        print("\n" + "="*50)
        print("🤖 LLM Processing Options:")
        print("1. Generate error SQLs and explanations")
        print("2. Skip LLM processing for now")
        
        # 这里可以添加用户选择逻辑
        use_llm = True  # 可以改为用户输入
        
        if use_llm:
            # 初始化LLM客户端
            try:
                
                # 方式1: 从环境变量读取API key
                # export OPENAI_API_KEY="your-api-key"
                client = openai.OpenAI()
                
                # 方式2: 直接设置API key (不推荐在生产环境)
                # client = openai.OpenAI(api_key="your-api-key-here")
                
                # 方式3: 使用其他LLM服务 (如Azure OpenAI)
                # client = openai.AzureOpenAI(
                #     azure_endpoint="https://your-resource.openai.azure.com/",
                #     api_key="your-api-key",
                #     api_version="2024-02-15-preview"
                # )
                
                llm_injector = LLMErrorInjector(llm_client=client)
                
            except ImportError:
                print("❌ OpenAI library not installed. Install with: pip install openai")
                print("⏭️ Skipping LLM processing.")
                return
            except Exception as e:
                print(f"❌ Failed to initialize LLM client: {e}")
                print("💡 Make sure to set OPENAI_API_KEY environment variable")
                print("⏭️ Skipping LLM processing.")
                return
            
            category_dir = os.path.join(output_dir, target_category)
            
            print(f"🔄 Processing prompts with LLM...")
            try:
                # 方式1: 按错误类型组织文件（推荐）
                generated_records = llm_injector.process_category_with_llm(
                    category_dir=category_dir,
                    organize_by_error_type=True
                )
                
                # 方式2: 简单处理（所有文件放一起）
                # llm_injector.process_prompts_in_directory(category_dir)
                
            except Exception as e:
                print(f"⚠️ Error during LLM processing: {e}")
        else:
            print("⏭️ Skipping LLM processing. Prompts are ready for manual LLM calls.")
        
    else:
        print(f"❌ Category '{target_category}' not found!")


def interactive_category_selection():
    """交互式类别选择"""
    # 你可以用这个函数来手动选择类别
    processor = ErrorInjectionProcessor(
        error_explanation_path="Spider/error_explaination.json",
        schema_root="spider2-lite/resource/databases", 
        sql_root="Spider/lite_256_true_sql",
        output_dir="Error_injection/Error_data"
    )
    
    categories = processor.list_available_categories()
    
    print("🎯 Select an error category to process:")
    for i, category in enumerate(categories, 1):
        print(f"  {i}. {category}")
    
    try:
        choice = int(input("\nEnter category number: ")) - 1
        if 0 <= choice < len(categories):
            selected_category = categories[choice]
            print(f"\n✅ Selected: {selected_category}")
            
            # 询问样本数量
            sample_size = int(input("Enter sample size per sub-error (default 10): ") or "10")
            
            # 询问是否使用LLM处理
            use_llm_input = input("Generate error SQLs with LLM? (y/n, default n): ").lower()
            use_llm = use_llm_input in ['y', 'yes']
            
            # 开始处理
            processor.generate_prompts_for_category(
                category_name=selected_category,
                jsonl_path="Spider/spider2-lite_with_golden.jsonl",
                sample_size=sample_size
            )
            
            if use_llm:
                print("\n🤖 Processing with LLM...")
                
                try:
                    client = openai.OpenAI()  # 从环境变量读取API key
                    llm_injector = LLMErrorInjector(llm_client=client)
                    
                except ImportError:
                    print("❌ OpenAI library not installed. Install with: pip install openai")
                    return
                except Exception as e:
                    print(f"❌ Failed to initialize LLM client: {e}")
                    print("💡 Make sure to set OPENAI_API_KEY environment variable")
                    return
                
                category_dir = os.path.join("Error_injection/Error_data", selected_category)
                
                try:
                    organize_files = input("Organize files by error type? (y/n, default y): ").lower()
                    organize_by_type = organize_files != 'n'
                    
                    llm_injector.process_category_with_llm(
                        category_dir=category_dir,
                        organize_by_error_type=organize_by_type
                    )
                except Exception as e:
                    print(f"❌ Error during LLM processing: {e}")
            else:
                print("✅ Prompts generated. You can process them with LLM later.")
        else:
            print("❌ Invalid choice!")
    except ValueError:
        print("❌ Please enter a valid number!")


if __name__ == "__main__":
    # 可以选择运行哪个函数
    # main()  # 自动处理
    interactive_category_selection()  # 交互式选择