import json
import sqlglot
from sqlglot import parse_one
import os

class LLMFriendlyProcessor:
    def __init__(self, default_dialect='sqlite'):
        self.default_dialect = default_dialect
        self.processed_count = 0
        self.error_count = 0
        self.errors = []
    
    def extract_semantic_info(self, ast_dict):
        """提取对LLM有用的语义信息"""
        result = {
            'query_type': ast_dict.get('class', 'Unknown'),
            'tables': [],
            'columns': [],
            'conditions': [],
            'aggregations': [],
            'window_functions': [],
            'joins': [],
            'subqueries': [],
            'grouping': [],
            'ordering': [],
            'complexity_score': 0
        }
        
        self._analyze_node(ast_dict, result)
        result['complexity_score'] = self._calculate_complexity(result)
        
        return result
    
    def _analyze_node(self, obj, result, context=''):
        """递归分析AST节点"""
        if not isinstance(obj, dict) or 'class' not in obj:
            return
        
        node_type = obj['class']
        args = obj.get('args', {})
        
        # 分析不同类型的节点
        if node_type == 'Select':
            self._analyze_select(args, result)
        elif node_type == 'Table':
            self._analyze_table(args, result)
        elif node_type == 'Column':
            self._analyze_column(args, result, context)
        elif node_type == 'Join':
            self._analyze_join(args, result)
        elif node_type in ['EQ', 'GT', 'LT', 'GTE', 'LTE', 'NEQ', 'Like', 'In']:
            self._analyze_condition(node_type, args, result)
        elif node_type in ['Count', 'Sum', 'Avg', 'Max', 'Min']:
            self._analyze_aggregation(node_type, args, result)
        elif node_type == 'Window':
            self._analyze_window_function(args, result)
        elif node_type == 'Subquery':
            result['subqueries'].append('nested_query')
        elif node_type == 'Group':
            self._analyze_group_by(args, result)
        elif node_type == 'Order':
            self._analyze_order_by(args, result)
        
        # 递归处理子节点
        for key, value in args.items():
            if isinstance(value, dict):
                self._analyze_node(value, result, key)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._analyze_node(item, result, key)
    
    def _analyze_select(self, args, result):
        """分析SELECT子句"""
        if 'expressions' in args:
            expressions = args['expressions']
            if isinstance(expressions, list):
                for expr in expressions:
                    if isinstance(expr, dict):
                        # 分析选择的列
                        if expr.get('class') == 'Column':
                            col_info = self._extract_column_info(expr)
                            if col_info:
                                result['columns'].append(col_info)
                        elif expr.get('class') == 'Alias':
                            # 处理别名
                            alias_name = self._get_string_value(expr.get('args', {}).get('alias'))
                            inner = expr.get('args', {}).get('this', {})
                            if inner.get('class') in ['Count', 'Sum', 'Avg', 'Max', 'Min']:
                                self._analyze_aggregation(inner.get('class'), inner.get('args', {}), result, alias_name)
    
    def _analyze_table(self, args, result):
        """分析表信息"""
        table_name = self._get_string_value(args.get('this'))
        alias = self._get_string_value(args.get('alias'))
        
        table_info = {'name': table_name}
        if alias:
            table_info['alias'] = alias
        
        if table_info not in result['tables']:
            result['tables'].append(table_info)
    
    def _analyze_column(self, args, result, context):
        """分析列信息"""
        if context in ['expressions', 'select']:  # 只在SELECT中记录列
            col_info = self._extract_column_info({'class': 'Column', 'args': args})
            if col_info and col_info not in result['columns']:
                result['columns'].append(col_info)
    
    def _extract_column_info(self, column_obj):
        """提取列信息"""
        args = column_obj.get('args', {})
        col_name = self._get_string_value(args.get('this'))
        table_ref = self._get_string_value(args.get('table'))
        
        if col_name:
            col_info = {'name': col_name}
            if table_ref:
                col_info['table'] = table_ref
            return col_info
        return None
    
    def _analyze_condition(self, operator, args, result):
        """分析WHERE条件"""
        left = args.get('this')
        right = args.get('expression')
        
        condition = {
            'operator': operator.lower(),
            'left': self._describe_expression(left),
            'right': self._describe_expression(right)
        }
        
        result['conditions'].append(condition)
    
    def _analyze_aggregation(self, func_type, args, result, alias=None):
        """分析聚合函数"""
        target = self._describe_expression(args.get('this'))
        
        agg_info = {
            'function': func_type.upper(),
            'target': target
        }
        
        if alias:
            agg_info['alias'] = alias
        
        result['aggregations'].append(agg_info)
    
    def _analyze_window_function(self, args, result):
        """分析窗口函数"""
        func = args.get('this', {})
        func_name = func.get('class', 'UNKNOWN') if isinstance(func, dict) else str(func)
        
        window_info = {
            'function': func_name.upper(),
            'has_partition': 'partition_by' in args,
            'has_order': 'order' in args
        }
        
        if 'partition_by' in args:
            window_info['partition_columns'] = self._extract_column_list(args['partition_by'])
        
        if 'order' in args:
            window_info['order_columns'] = self._extract_column_list(args['order'])
        
        result['window_functions'].append(window_info)
    
    def _analyze_join(self, args, result):
        """分析JOIN"""
        join_table = self._describe_expression(args.get('this'))
        join_condition = self._describe_expression(args.get('on'))
        
        join_info = {
            'table': join_table,
            'condition': join_condition,
            'type': 'INNER'  # 默认，可以进一步解析join类型
        }
        
        result['joins'].append(join_info)
    
    def _analyze_group_by(self, args, result):
        """分析GROUP BY"""
        if 'expressions' in args:
            for expr in args['expressions']:
                col_desc = self._describe_expression(expr)
                if col_desc:
                    result['grouping'].append(col_desc)
    
    def _analyze_order_by(self, args, result):
        """分析ORDER BY"""
        if 'expressions' in args:
            for expr in args['expressions']:
                col_desc = self._describe_expression(expr)
                if col_desc:
                    result['ordering'].append(col_desc)
    
    def _describe_expression(self, expr):
        """描述表达式"""
        if isinstance(expr, dict) and 'class' in expr:
            node_type = expr['class']
            args = expr.get('args', {})
            
            if node_type == 'Column':
                col_name = self._get_string_value(args.get('this'))
                table_ref = self._get_string_value(args.get('table'))
                return f"{table_ref}.{col_name}" if table_ref else col_name
            elif node_type == 'Literal':
                value = args.get('this')
                if args.get('is_string'):
                    return f"'{value}'"
                return str(value)
            elif node_type == 'Identifier':
                return self._get_string_value(args.get('this'))
            elif node_type in ['Count', 'Sum', 'Avg', 'Max', 'Min']:
                target = self._describe_expression(args.get('this'))
                return f"{node_type.upper()}({target})"
            else:
                return node_type
        
        return str(expr) if expr else None
    
    def _extract_column_list(self, expr_list):
        """提取列列表"""
        if isinstance(expr_list, list):
            return [self._describe_expression(expr) for expr in expr_list]
        return [self._describe_expression(expr_list)] if expr_list else []
    
    def _get_string_value(self, obj):
        """安全获取字符串值"""
        if isinstance(obj, dict) and 'class' in obj:
            if obj['class'] == 'Identifier':
                return obj.get('args', {}).get('this')
        return str(obj) if obj else None
    
    def _calculate_complexity(self, result):
        """计算查询复杂度分数"""
        score = 0
        score += len(result['tables']) * 1
        score += len(result['joins']) * 2
        score += len(result['subqueries']) * 3
        score += len(result['window_functions']) * 2
        score += len(result['aggregations']) * 1
        score += len(result['conditions']) * 1
        
        return score
    
    def parse_sql_to_ast(self, sql: str, dialect: str = None):
        """解析SQL为LLM友好的格式"""
        try:
            ast = parse_one(sql, dialect=dialect or self.default_dialect)
            
            # 获取AST字典
            if hasattr(ast, 'to_dict'):
                ast_dict = ast.to_dict()
            elif hasattr(ast, 'dump'):
                ast_dump = ast.dump()
                if isinstance(ast_dump, str):
                    ast_dict = eval(ast_dump)
                else:
                    ast_dict = ast_dump
            else:
                return {'error': 'Cannot convert AST to dict'}
            
            # 提取语义信息
            semantic_info = self.extract_semantic_info(ast_dict)
            
            # 添加原始SQL用于参考
            semantic_info['original_sql'] = sql
            
            return semantic_info
                
        except Exception as e:
            self.error_count += 1
            return {'error': str(e), 'original_sql': sql}
    
    def process_json_file(self, input_file: str, output_file: str, 
                         sql_field: str = 'sql', 
                         new_field: str = 'ast',
                         keep_original: bool = True):
        """批量处理JSON文件"""
        
        print(f"处理文件: {input_file} (LLM友好格式)")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and sql_field in item:
                    sql = item[sql_field]
                    ast = self.parse_sql_to_ast(sql)
                    item[new_field] = ast
                    
                    if not keep_original:
                        del item[sql_field]
                    
                    self.processed_count += 1
                    
                    if (i + 1) % 100 == 0:
                        print(f"已处理 {i + 1} 条...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        success_count = self.processed_count - self.error_count
        print(f"完成! 处理:{self.processed_count} 成功:{success_count} 失败:{self.error_count}")
        
        return {'processed': self.processed_count, 'success': success_count, 'errors': self.error_count}

def test_llm_friendly():
    """测试LLM友好的输出"""
    test_cases = [
        "SELECT name FROM users WHERE age > 18",
        "SELECT u.name, COUNT(o.id) as order_count FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
        "SELECT name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rank FROM employees",
        "SELECT name, CASE WHEN age > 18 THEN 'adult' ELSE 'minor' END FROM users"
    ]
    
    processor = LLMFriendlyProcessor()
    
    for i, sql in enumerate(test_cases, 1):
        print(f"\n=== 测试 {i} ===")
        print(f"SQL: {sql}")
        
        ast = processor.parse_sql_to_ast(sql)
        print("LLM友好的AST:")
        print(json.dumps(ast, indent=2, ensure_ascii=False))

def main():
    print("LLM友好的SQL AST处理器")
    print("=" * 50)
    
    test_llm_friendly()
    
    print("\n" + "=" * 50)
    print("这个版本为LLM提供了:")
    print("- 清晰的查询意图和结构")
    print("- 详细的表和列信息")  
    print("- 具体的条件和聚合逻辑")
    print("- 复杂度评分")
    print("- 语义化的描述")

if __name__ == "__main__":
    main()