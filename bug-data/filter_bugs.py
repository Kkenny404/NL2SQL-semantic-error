import json
import sqlglot
from sqlglot import parse_one
import os

class UltraSimpleSQLProcessor:
    def __init__(self, default_dialect='sqlite'):
        self.default_dialect = default_dialect
        self.processed_count = 0
        self.error_count = 0
        self.errors = []
    
    def clean_ast(self, obj):
        """递归清理AST，移除meta、typed、safe等无用字段"""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                # 跳过无用字段
                if key in ['meta', 'typed', 'safe', 'parent']:
                    continue
                cleaned[key] = self.clean_ast(value)
            return cleaned
        elif isinstance(obj, list):
            return [self.clean_ast(item) for item in obj]
        else:
            return obj
    
    def parse_sql_to_ast(self, sql: str, dialect: str = None):
        """解析SQL为AST，移除所有无用信息"""
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
            
            # 清理无用字段
            return self.clean_ast(ast_dict)
                
        except Exception as e:
            self.error_count += 1
            error_info = {
                'sql': sql[:80] + '...' if len(sql) > 80 else sql,
                'error': str(e)
            }
            self.errors.append(error_info)
            return {'error': str(e)}
    
    def process_json_file(self, input_file: str, output_file: str, 
                         sql_field: str = 'sql', 
                         new_field: str = 'ast',
                         keep_original: bool = True):
        """批量处理JSON文件"""
        
        print(f"处理文件: {input_file}")
        
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
        
        elif isinstance(data, dict):
            if sql_field in data:
                sql = data[sql_field]
                ast = self.parse_sql_to_ast(sql)
                data[new_field] = ast
                
                if not keep_original:
                    del data[sql_field]
                
                self.processed_count += 1
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        success_count = self.processed_count - self.error_count
        success_rate = (success_count / self.processed_count * 100) if self.processed_count > 0 else 0
        
        print(f"完成! 处理:{self.processed_count} 成功:{success_count} 失败:{self.error_count} 成功率:{success_rate:.1f}%")
        
        return {
            'processed': self.processed_count,
            'success': success_count,
            'errors': self.error_count,
            'rate': success_rate
        }

def test_comparison():
    """对比清理前后的差异"""
    sql = "SELECT MAX(CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) FROM frpm WHERE `County Name` = 'Alameda'"
    
    processor = UltraSimpleSQLProcessor()
    
    print("=== 原始SQL ===")
    print(sql)
    
    print("\n=== 清理后的AST ===")
    clean_ast = processor.parse_sql_to_ast(sql)
    print(json.dumps(clean_ast, indent=2, ensure_ascii=False))
    
    print(f"\n=== 大小对比 ===")
    original_size = len(json.dumps(clean_ast).encode('utf-8'))
    print(f"清理后大小: {original_size} 字节")

def main():
    print("超简化SQL AST处理器 - 移除所有无用信息")
    print("=" * 50)
    
    # 测试
    test_comparison()
    
    print("\n" + "=" * 50)
    
    # 文件处理
    INPUT_FILE = "bug-data/NL2SQL-Bugs.json"  # 改为你的文件路径
    OUTPUT_FILE = "bug-data/NL2SQL-Bugs-with-AST.json"
    
    if os.path.exists(INPUT_FILE):
        processor = UltraSimpleSQLProcessor()
        stats = processor.process_json_file(INPUT_FILE, OUTPUT_FILE)
        print(f"输出: {OUTPUT_FILE}")
    else:
        print(f"请将 INPUT_FILE 改为你的实际文件路径")
        print("示例用法:")
        print("processor = UltraSimpleSQLProcessor()")
        print("processor.process_json_file('input.json', 'output.json')")

if __name__ == "__main__":
    main()