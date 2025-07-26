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

class AggregateFilterStrategy(FilterStrategy):
    """筛选包含聚合函数的 SQL"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        aggregate_keywords = ["SUM", "COUNT", "AVG", "MAX", "MIN", "GROUP BY", "HAVING"]
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                if any(keyword in sql_content for keyword in aggregate_keywords):
                    filtered_entries.append(entry)
            except FileNotFoundError:
                continue
        return filtered_entries

class SubqueryFilterStrategy(FilterStrategy):
    """筛选包含子查询的 SQL"""
    
    def filter(self, dataset: List[Dict], sql_root: str) -> List[Dict]:
        filtered_entries = []
        for entry in dataset:
            sql_id = entry["instance_id"]
            try:
                sql_content = load_sql(sql_root, sql_id).upper()
                # 简单检测：包含 SELECT 且有嵌套括号结构
                if sql_content.count("SELECT") > 1 or ("EXISTS" in sql_content) or ("IN (" in sql_content):
                    filtered_entries.append(entry)
            except FileNotFoundError:
                continue
        return filtered_entries



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
                "Table Confusion": NoFilterStrategy(),
            },
            "Value-related Errors": {
                "Value Mismatch": ValueMismatchFilterStrategy(),
                "Data Format Mismatch": DataFormatMismatchFilterStrategy(),
            },
            "Operator-related Errors": {
                "Comparison Operator": ComparisonOperatorFilterStrategy(),
                "Logical Operator": LogicalOperatorFilterStrategy(),
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