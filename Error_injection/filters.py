# ========== 过滤器策略模式 ==========

from abc import ABC, abstractmethod
from typing import Dict, List

from share_utils import load_sql


class FilterStrategy(ABC):
    """过滤器策略抽象基类"""
    
    @abstractmethod
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        """过滤数据集"""
        pass

class NoFilterStrategy(FilterStrategy):
    """无过滤策略"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        return dataset

class JoinFilterStrategy(FilterStrategy):
    """筛选包含 JOIN 的 SQL"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        join_sql_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id)
                if "JOIN" in sql_content.upper():
                    join_sql_entries.append(entry)
            except FileNotFoundError:
                continue
        return join_sql_entries


class ValueMismatchFilterStrategy(FilterStrategy):
    """筛选包含具体值比较的 SQL - 适用于 Value Mismatch 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 放宽条件：检查是否包含任何值操作
                has_value_operations = (
                    # 基本比较操作
                    "=" in sql_content or 
                    ">" in sql_content or 
                    "<" in sql_content or
                    ">=" in sql_content or
                    "<=" in sql_content or
                    "!=" in sql_content or
                    "<>" in sql_content or
                    # 模式匹配和范围
                    "LIKE" in sql_content or
                    "IN (" in sql_content or
                    "BETWEEN" in sql_content or
                    # WHERE子句通常包含值比较
                    "WHERE" in sql_content or
                    # HAVING子句也包含值比较
                    "HAVING" in sql_content
                )
                
                if has_value_operations:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class DataFormatMismatchFilterStrategy(FilterStrategy):
    """筛选包含日期、时间或特定格式数据的 SQL - 适用于 Data Format Mismatch 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 放宽条件：检查是否可能包含格式敏感的操作
                has_format_operations = (
                    # 日期时间相关关键词
                    "DATE" in sql_content or
                    "TIME" in sql_content or
                    "YEAR" in sql_content or
                    "MONTH" in sql_content or
                    "DAY" in sql_content or
                    # 类型转换函数
                    "CAST" in sql_content or
                    "CONVERT" in sql_content or
                    # 可能的日期格式模式（更宽松）
                    "-" in sql_content or  # 可能是日期分隔符
                    "/" in sql_content or  # 可能是日期分隔符
                    # 数字相关（可能涉及格式问题）
                    any(char.isdigit() for char in sql_content) or
                    # 字符串操作（可能涉及格式转换）
                    "'" in sql_content or
                    '"' in sql_content
                )
                
                if has_format_operations:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries

# ======================================
# ========== Operation Related ==========
# =======================================

class ComparisonOperatorFilterStrategy(FilterStrategy):
    """筛选包含比较运算符的 SQL - 适用于 Comparison Operator 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查是否包含比较运算符
                comparison_operators = [
                    "=", "!=", "<>", ">", "<", ">=", "<=",
                    "LIKE", "NOT LIKE", "ILIKE", "NOT ILIKE",
                    "IN", "NOT IN", "EXISTS", "NOT EXISTS",
                    "IS NULL", "IS NOT NULL",
                    "BETWEEN", "NOT BETWEEN",
                    "REGEXP", "NOT REGEXP", "RLIKE"
                ]
                
                has_comparison_operators = any(
                    op in sql_content for op in comparison_operators
                )
                
                # 额外检查：确保有WHERE子句或HAVING子句（比较运算符通常出现在这里）
                has_conditional_clause = (
                    "WHERE" in sql_content or 
                    "HAVING" in sql_content or
                    "ON" in sql_content  # JOIN条件中也可能有比较运算符
                )
                
                if has_comparison_operators and has_conditional_clause:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class LogicalOperatorFilterStrategy(FilterStrategy):
    """筛选包含逻辑运算符的 SQL - 适用于 Logical Operator 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查是否包含逻辑运算符
                logical_operators = ["AND", "OR", "NOT"]
                
                has_logical_operators = any(
                    op in sql_content for op in logical_operators
                )
                
                # 检查是否有多个条件（逻辑运算符通常用于连接多个条件）
                has_multiple_conditions = (
                    sql_content.count("AND") > 0 or 
                    sql_content.count("OR") > 0 or
                    sql_content.count("NOT") > 0
                )
                
                # 检查是否有WHERE或HAVING子句（逻辑运算符主要出现在条件子句中）
                has_conditional_clause = (
                    "WHERE" in sql_content or 
                    "HAVING" in sql_content
                )
                
                if has_logical_operators and has_multiple_conditions and has_conditional_clause:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class ConditionRelatedFilterStrategy(FilterStrategy):
    """筛选包含条件相关的 SQL - 适用于所有 Condition-related 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查是否包含条件子句
                has_conditional_clause = (
                    "WHERE" in sql_content or 
                    "HAVING" in sql_content or
                    "ON" in sql_content  # JOIN条件也可能有condition错误
                )
                
                # 检查是否包含条件相关的关键词
                condition_keywords = [
                    # 比较运算符
                    "=", "!=", "<>", ">", "<", ">=", "<=",
                    # 模式匹配和集合
                    "LIKE", "IN", "EXISTS", "BETWEEN",
                    # NULL处理（重要：针对implicit condition）
                    "IS NULL", "IS NOT NULL", "ISNULL", "COALESCE", "NULLIF",
                    # 逻辑运算符
                    "AND", "OR", "NOT"
                ]
                
                has_condition_operators = any(
                    keyword in sql_content for keyword in condition_keywords
                )
                
                # 额外检查：是否有可能的条件表达式结构
                # 寻找 column = value 或类似的模式
                has_potential_conditions = (
                    # 基本的条件结构指示符
                    "." in sql_content or  # table.column 格式
                    "'" in sql_content or  # 字符串值
                    any(char.isdigit() for char in sql_content) or  # 数字值
                    # 常见的条件模式
                    " = " in sql_content or
                    " > " in sql_content or
                    " < " in sql_content
                )
                
                # 如果有条件子句且包含条件操作符，则认为可能存在条件相关错误
                if has_conditional_clause and (has_condition_operators or has_potential_conditions):
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class AggregateFunctionFilterStrategy(FilterStrategy):
    """筛选包含聚合函数的 SQL - 适用于 Aggregate Functions 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查聚合函数关键词
                aggregate_keywords = [
                    "SUM", "COUNT", "AVG", "MAX", "MIN", 
                    "GROUP BY", "HAVING", "DISTINCT"
                ]
                
                has_aggregate_functions = any(
                    keyword in sql_content for keyword in aggregate_keywords
                )
                
                if has_aggregate_functions:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class WindowFunctionFilterStrategy(FilterStrategy):
    """筛选包含窗口函数的 SQL - 适用于 Window Functions 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查窗口函数关键词
                window_keywords = [
                    "OVER", "PARTITION BY", "ROW_NUMBER", "RANK", "DENSE_RANK",
                    "LEAD", "LAG", "FIRST_VALUE", "LAST_VALUE", "NTILE"
                ]
                
                has_window_functions = any(
                    keyword in sql_content for keyword in window_keywords
                )
                
                if has_window_functions:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class DateTimeFunctionFilterStrategy(FilterStrategy):
    """筛选包含日期时间函数的 SQL - 适用于 Date/Time Functions 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查日期时间函数关键词
                datetime_keywords = [
                    "DATE", "TIME", "DATETIME", "TIMESTAMP",
                    "STRFTIME", "DATEADD", "DATEDIFF", "DATEPART",
                    "NOW", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
                    "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND",
                    "TO_DATE", "TO_TIMESTAMP", "EXTRACT"
                ]
                
                has_datetime_functions = any(
                    keyword in sql_content for keyword in datetime_keywords
                )
                
                if has_datetime_functions:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class ConversionFunctionFilterStrategy(FilterStrategy):
    """筛选包含转换函数的 SQL - 适用于 Conversion Functions 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查转换函数关键词
                conversion_keywords = [
                    "CAST", "CONVERT", "TRY_CAST", "TRY_CONVERT",
                    "TO_CHAR", "TO_NUMBER", "TO_DATE", "TO_TIMESTAMP",
                    "PARSE", "TRY_PARSE", "FORMAT"
                ]
                
                has_conversion_functions = any(
                    keyword in sql_content for keyword in conversion_keywords
                )
                
                if has_conversion_functions:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class MathFunctionFilterStrategy(FilterStrategy):
    """筛选包含数学函数的 SQL - 适用于 Math Functions 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查数学函数关键词
                math_keywords = [
                    "ROUND", "CEIL", "CEILING", "FLOOR", "TRUNC", "TRUNCATE",
                    "ABS", "SQRT", "POWER", "EXP", "LOG", "LN",
                    "SIN", "COS", "TAN", "MOD", "RAND", "RANDOM"
                ]
                
                has_math_functions = any(
                    keyword in sql_content for keyword in math_keywords
                )
                
                if has_math_functions:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class StringFunctionFilterStrategy(FilterStrategy):
    """筛选包含字符串函数的 SQL - 适用于 String Functions 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查字符串函数关键词
                string_keywords = [
                    "SUBSTR", "SUBSTRING", "CONCAT", "LENGTH", "LEN",
                    "TRIM", "LTRIM", "RTRIM", "UPPER", "LOWER",
                    "REPLACE", "TRANSLATE", "REVERSE", "LEFT", "RIGHT",
                    "CHARINDEX", "INSTR", "POSITION", "LOCATE",
                    "SPLIT", "STRING_AGG", "STUFF", "REPLICATE"
                ]
                
                has_string_functions = any(
                    keyword in sql_content for keyword in string_keywords
                )
                
                if has_string_functions:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class ConditionalFunctionFilterStrategy(FilterStrategy):
    """筛选包含条件函数的 SQL - 适用于 Conditional Functions 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查条件函数关键词
                conditional_keywords = [
                    "CASE WHEN", "CASE", "WHEN", "THEN", "ELSE", "END",
                    "IIF", "ISNULL", "NULLIF", "COALESCE", "NVL", "NVL2",
                    "DECODE", "GREATEST", "LEAST"
                ]
                
                has_conditional_functions = any(
                    keyword in sql_content for keyword in conditional_keywords
                )
                
                if has_conditional_functions:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries
    
class ClauseRelatedFilterStrategy(FilterStrategy):
    """筛选包含分组、排序或聚合子句的 SQL - 适用于 Clause-related Errors"""

    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()

                # 检查是否有 GROUP BY / ORDER BY / HAVING
                clause_keywords = ["GROUP BY", "ORDER BY", "HAVING"]
                has_relevant_clauses = any(keyword in sql_content for keyword in clause_keywords)

                if has_relevant_clauses:
                    filtered_entries.append(entry)

            except FileNotFoundError:
                continue
        return filtered_entries

class SubqueryStrategy(FilterStrategy):
    """筛选包含子查询的 SQL - 用于 Subquery Missing 和 Mismatch 错误"""
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            try:
                sql_content = load_sql(sql_root, entry["instance_id"]).upper()
                has_subquery = (
                    "SELECT" in sql_content and "(" in sql_content and "SELECT" in sql_content.split("(")[-1]
                ) or any(k in sql_content for k in ["IN (", "EXISTS (", "ANY (", "SOME ("])
                if has_subquery:
                    filtered_entries.append(entry)
            except FileNotFoundError:
                continue
        return filtered_entries


class AscDescFilterStrategy(FilterStrategy):
    """筛选包含排序的 SQL - 适用于 ASC/DESC 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查是否包含排序相关的关键词
                order_keywords = [
                    "ORDER BY", "SORT BY",
                    "ASC", "DESC", 
                    "ASCENDING", "DESCENDING"
                ]
                
                # 检查可能需要排序的场景
                order_scenarios = [
                    "LIMIT", "TOP", "FIRST", "LAST",
                    "RANK", "ROW_NUMBER", "DENSE_RANK",
                    "HIGHEST", "LOWEST", "MAXIMUM", "MINIMUM"
                ]
                
                has_order_keywords = any(
                    keyword in sql_content for keyword in order_keywords
                )
                
                has_order_scenarios = any(
                    scenario in sql_content for scenario in order_scenarios
                )
                
                # 基本SQL结构检查
                has_basic_structure = (
                    "SELECT" in sql_content and 
                    "FROM" in sql_content
                )
                
                if has_basic_structure and (has_order_keywords or has_order_scenarios):
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class DistinctFilterStrategy(FilterStrategy):
    """筛选可能涉及去重的 SQL - 适用于 DISTINCT 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查去重相关关键词
                distinct_keywords = [
                    "DISTINCT", "UNIQUE", "DEDUPLICATE"
                ]
                
                # 检查可能需要去重的场景
                dedup_scenarios = [
                    "GROUP BY", "UNION", "UNION ALL",
                    "COUNT(DISTINCT", "SUM(DISTINCT", "AVG(DISTINCT",
                    "DUPLICATE", "REMOVE", "ELIMINATE"
                ]
                
                # 检查聚合函数（可能需要DISTINCT修饰符）
                aggregate_functions = [
                    "COUNT(", "SUM(", "AVG(", "MAX(", "MIN("
                ]
                
                has_distinct_keywords = any(
                    keyword in sql_content for keyword in distinct_keywords
                )
                
                has_dedup_scenarios = any(
                    scenario in sql_content for scenario in dedup_scenarios
                )
                
                has_aggregate_functions = any(
                    func in sql_content for func in aggregate_functions
                )
                
                # 基本SQL结构检查
                has_basic_structure = (
                    "SELECT" in sql_content and 
                    "FROM" in sql_content
                )
                
                if (has_basic_structure and 
                    (has_distinct_keywords or has_dedup_scenarios or has_aggregate_functions)):
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class OthersFilterStrategy(FilterStrategy):
    """筛选复杂查询 SQL - 适用于 Others 错误（SQL意图严重偏离）"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查复杂SQL结构（更容易出现严重偏离错误）
                complex_structures = [
                    "WITH", "CTE", "RECURSIVE",
                    "UNION", "INTERSECT", "EXCEPT",
                    "EXISTS", "NOT EXISTS",
                    "CASE WHEN", "IF", "IIF",
                    "SUBQUERY", "NESTED"
                ]
                
                # 检查多表操作
                multi_table_operations = [
                    "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
                    "CROSS JOIN", "NATURAL JOIN",
                    "FROM.*,.*FROM"  # 多个FROM子句的模式
                ]
                
                # 检查高级功能
                advanced_features = [
                    "WINDOW", "OVER", "PARTITION BY",
                    "PIVOT", "UNPIVOT",
                    "LATERAL", "APPLY",
                    "MERGE", "UPSERT"
                ]
                
                # 计算SQL复杂度指标
                complexity_indicators = [
                    sql_content.count("SELECT"),  # 多个SELECT表示子查询
                    sql_content.count("("),       # 括号数量
                    sql_content.count("JOIN"),    # JOIN数量
                    sql_content.count("AND"),     # 条件复杂度
                    sql_content.count("OR"),      # 条件复杂度
                ]
                
                has_complex_structures = any(
                    struct in sql_content for struct in complex_structures
                )
                
                has_multi_table_ops = any(
                    op in sql_content for op in multi_table_operations
                )
                
                has_advanced_features = any(
                    feature in sql_content for feature in advanced_features
                )
                
                # 复杂度评分：如果满足多个条件或单个指标较高，认为是复杂查询
                complexity_score = sum(complexity_indicators)
                is_complex = (
                    complexity_score > 10 or  # 总体复杂度高
                    sql_content.count("SELECT") > 2 or  # 多层嵌套
                    (has_complex_structures and has_multi_table_ops) or  # 结构复杂且多表
                    has_advanced_features  # 使用高级功能
                )
                
                # 基本SQL结构检查
                has_basic_structure = (
                    "SELECT" in sql_content and 
                    "FROM" in sql_content
                )
                
                if has_basic_structure and is_complex:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class JoinConditionMismatchFilterStrategy(FilterStrategy):
    """筛选包含JOIN条件的 SQL - 适用于 Join Condition Mismatch 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查JOIN相关关键词
                join_keywords = [
                    "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
                    "CROSS JOIN", "NATURAL JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN",
                    "FULL OUTER JOIN"
                ]
                
                # 检查JOIN条件关键词
                join_condition_keywords = [
                    " ON ", " USING", "=", "AND", "OR"
                ]
                
                # 检查是否有多表查询（FROM子句中的逗号连接也算）
                multi_table_indicators = [
                    "FROM.*,.*",  # FROM table1, table2 pattern
                    "AS ", "TABLE", "VIEW"
                ]
                
                has_join_keywords = any(
                    keyword in sql_content for keyword in join_keywords
                )
                
                has_join_conditions = any(
                    keyword in sql_content for keyword in join_condition_keywords
                )
                
                # 检查是否可能有多表关系
                has_multi_tables = (
                    sql_content.count("FROM") >= 1 and
                    (sql_content.count(",") > 0 or has_join_keywords)
                )
                
                # 基本SQL结构检查
                has_basic_structure = (
                    "SELECT" in sql_content and 
                    "FROM" in sql_content
                )
                
                # 如果包含JOIN关键词，或者有多表且有条件，则可能存在JOIN条件错误
                if (has_basic_structure and 
                    (has_join_keywords or (has_multi_tables and has_join_conditions))):
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries


class JoinTypeMismatchFilterStrategy(FilterStrategy):
    """筛选包含不同JOIN类型的 SQL - 适用于 Join Type Mismatch 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查具体的JOIN类型
                specific_join_types = [
                    "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
                    "LEFT OUTER JOIN", "RIGHT OUTER JOIN", "FULL OUTER JOIN",
                    "CROSS JOIN", "NATURAL JOIN"
                ]
                
                # 检查通用JOIN（可能需要指定类型）
                general_join = "JOIN"
                
                # 检查可能暗示特定JOIN需求的关键词
                join_intent_keywords = [
                    "ALL", "OPTIONAL", "REQUIRED", "MATCH", "EXISTS",
                    "NULL", "NOT NULL", "MISSING", "EXCLUDE",
                    "INCLUDE", "BOTH", "EITHER", "ONLY"
                ]
                
                # 检查外键/主键相关词汇（通常暗示JOIN关系）
                key_relationship_keywords = [
                    "ID", "KEY", "FOREIGN", "PRIMARY", "REFERENCE",
                    "LINK", "CONNECT", "RELATE", "ASSOCIATE"
                ]
                
                has_specific_join_types = any(
                    join_type in sql_content for join_type in specific_join_types
                )
                
                has_general_join = general_join in sql_content
                
                has_join_intent = any(
                    keyword in sql_content for keyword in join_intent_keywords
                )
                
                has_key_relationships = any(
                    keyword in sql_content for keyword in key_relationship_keywords
                )
                
                # 基本SQL结构检查
                has_basic_structure = (
                    "SELECT" in sql_content and 
                    "FROM" in sql_content
                )
                
                # 如果包含JOIN类型，或者有JOIN意图提示，则可能存在JOIN类型错误
                if (has_basic_structure and 
                    (has_specific_join_types or has_general_join or 
                     (has_join_intent and (has_key_relationships or sql_content.count(",") > 0)))):
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries



class AttributeRelatedFilterStrategy(FilterStrategy):
    """筛选包含属性操作的 SQL - 适用于所有 Attribute-related 错误"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                
                # 检查SELECT子句（属性选择的主要位置）
                has_select_clause = "SELECT" in sql_content
                
                # 检查是否有多个属性/列的选择
                column_indicators = [
                    ",",  # 多列分隔符
                    ".",  # table.column格式
                    " AS ", " AS\n", " AS\t",  # 列别名
                    "*",  # 全选
                ]
                
                has_multiple_columns = any(
                    indicator in sql_content for indicator in column_indicators
                )
                
                # 检查WHERE子句中的属性引用
                has_where_clause = "WHERE" in sql_content
                
                # 检查常见的属性操作模式
                attribute_patterns = [
                    # 列引用模式
                    "SELECT.*FROM",
                    "WHERE.*=", "WHERE.*>", "WHERE.*<",
                    "GROUP BY", "ORDER BY", "HAVING",
                    # 函数中的列引用
                    "COUNT(", "SUM(", "AVG(", "MAX(", "MIN(",
                    "DISTINCT", "UNIQUE"
                ]
                
                has_attribute_operations = any(
                    pattern in sql_content.replace(" ", "").replace("\n", "").replace("\t", "")
                    for pattern in attribute_patterns
                    if "*" not in pattern
                ) or any(
                    pattern in sql_content for pattern in attribute_patterns if "*" in pattern
                )
                
                # 检查可能的表和列结构
                table_column_indicators = [
                    "FROM", "JOIN", "ON", "USING",
                    "INSERT", "UPDATE", "DELETE"
                ]
                
                has_table_references = any(
                    indicator in sql_content for indicator in table_column_indicators
                )
                
                # 检查复杂的列操作（更容易出现属性错误）
                complex_column_operations = [
                    "CASE WHEN", "IF(", "IIF(",
                    "CAST(", "CONVERT(",
                    "SUBSTRING(", "SUBSTR(", "LEFT(", "RIGHT(",
                    "CONCAT(", "UPPER(", "LOWER(",
                    "DATE(", "YEAR(", "MONTH(", "DAY("
                ]
                
                has_complex_operations = any(
                    operation in sql_content for operation in complex_column_operations
                )
                
                # 基本SQL结构检查
                has_basic_structure = (
                    has_select_clause and 
                    has_table_references
                )
                
                # 评估是否可能存在属性相关错误
                is_attribute_relevant = (
                    has_basic_structure and 
                    (has_multiple_columns or has_where_clause or 
                     has_attribute_operations or has_complex_operations)
                )
                
                if is_attribute_relevant:
                    filtered_entries.append(entry)
                    
            except FileNotFoundError:
                continue
        return filtered_entries



# ========== 错误类别管理器 ==========

class ErrorCategoryManager:
    """错误类别管理器"""
    
    def __init__(self):
        # 定义每个错误类别对应的过滤策略
        self.category_filters = {
            "Table-related Errors": {
                "Join Condition Mismatch": JoinFilterStrategy(),
                "Join Type Mismatch": JoinFilterStrategy(),
                "Table Missing": NoFilterStrategy(),
                "Table Redundancy": NoFilterStrategy(),
                "Table Mismatch": NoFilterStrategy(),
            },
            "Value-related Errors": {
                "Value Mismatch": ValueMismatchFilterStrategy(),
                "Data Format Mismatch": DataFormatMismatchFilterStrategy(),
            },
            "Operator-related Errors": {
                "Comparison Operator": ComparisonOperatorFilterStrategy(),
                "Logical Operator": LogicalOperatorFilterStrategy(),
            },
            "Condition-related Errors": {
                "Explicit Condition Missing": ConditionRelatedFilterStrategy(),
                "Explicit Condition Mismatch": ConditionRelatedFilterStrategy(),
                "Explicit Condition Redundancy": ConditionRelatedFilterStrategy(),
                "Implicit Condition Missing": ConditionRelatedFilterStrategy(),
            },
            "Function-related Errors": {
                "Aggregate Functions": AggregateFunctionFilterStrategy(),
                "Window Functions": WindowFunctionFilterStrategy(),
                "DateTime Functions": DateTimeFunctionFilterStrategy(),
                "Conversion Functions": ConversionFunctionFilterStrategy(),
                "Math Functions": MathFunctionFilterStrategy(),
                "String Functions": StringFunctionFilterStrategy(),
                "Conditional Functions": ConditionalFunctionFilterStrategy(),
            },
             "Clause-related Errors": {
                "Clause Missing": ClauseRelatedFilterStrategy(),
                "Clause Redundancy": NoFilterStrategy(),
            },
            "Subquery-related Errors": {
                "Subquery Missing": SubqueryStrategy(),
                "Subquery Mismatch": SubqueryStrategy(),
                "Partial Query": SubqueryStrategy(),
            },
            "Other Errors": {
                "ASC_DESC": AscDescFilterStrategy(),
                "DISTINCT": DistinctFilterStrategy(),
                "Others": OthersFilterStrategy(),
            },
            "Attribute-related Errors": {
                "Attribute Mismatch": AttributeRelatedFilterStrategy(),
                "Attribute Redundancy": AttributeRelatedFilterStrategy(),
                "Attribute Missing": AttributeRelatedFilterStrategy(),
            }
        }
    
    def get_filter_strategy(self, category: str, sub_error_type: str) -> FilterStrategy:
        """获取指定错误类型的过滤策略"""
        if category in self.category_filters:
            if sub_error_type in self.category_filters[category]:
                return self.category_filters[category][sub_error_type]
        return NoFilterStrategy()
    
    def get_available_categories(self) -> List[str]:
        """获取所有可用的错误类别"""
        return list(self.category_filters.keys())