import json
import os

def filter_json_by_instance_id(json_file_path, sql_folder_path, output_file_path):
    """
    过滤JSON文件，只保留instance_id在指定文件夹中存在对应文件的条目
    
    Args:
        json_file_path: 输入的JSON文件路径
        sql_folder_path: SQL文件夹路径
        output_file_path: 输出的JSON文件路径
    """
    
    # 获取SQL文件夹中的所有文件名（不包括扩展名）
    if not os.path.exists(sql_folder_path):
        print(f"错误: 文件夹 {sql_folder_path} 不存在")
        return
    
    sql_files = set()
    for filename in os.listdir(sql_folder_path):
        if filename.endswith('.sql'):
            # 移除.sql扩展名，获取文件名
            instance_name = filename[:-4]
            sql_files.add(instance_name)
    
    print(f"在文件夹中找到 {len(sql_files)} 个SQL文件")
    
    # 读取JSON文件
    if not os.path.exists(json_file_path):
        print(f"错误: JSON文件 {json_file_path} 不存在")
        return
    
    filtered_data = []
    total_count = 0
    filtered_count = 0
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data = json.loads(line)
                    total_count += 1
                    
                    # 检查instance_id是否在SQL文件中存在
                    instance_id = data.get('instance_id', '')
                    
                    if instance_id in sql_files:
                        filtered_data.append(data)
                        filtered_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"警告: 无法解析JSON行: {line[:50]}... 错误: {e}")
    
    # 写入过滤后的数据到新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for data in filtered_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"处理完成!")
    print(f"原始条目数: {total_count}")
    print(f"过滤后条目数: {filtered_count}")
    print(f"移除的条目数: {total_count - filtered_count}")
    print(f"输出文件: {output_file_path}")

# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    json_file = "spider2-lite/spider2-lite.jsonl"  # 输入的JSON文件
    sql_folder = "Spider/lite_256_true_sql"  # SQL文件夹路径
    output_file = "spider2-lite-filtered.jsonl"  # 输出的JSON文件
    
    # 执行过滤
    filter_json_by_instance_id(json_file, sql_folder, output_file)
    
    # 可选：显示一些被过滤掉的instance_id示例
    print("\n--- 额外信息 ---")
    
    # 获取SQL文件夹中的文件名
    sql_files = set()
    if os.path.exists(sql_folder):
        for filename in os.listdir(sql_folder):
            if filename.endswith('.sql'):
                sql_files.add(filename[:-4])
    
    # 找出JSON中存在但SQL文件夹中不存在的instance_id
    json_instance_ids = set()
    missing_ids = []
    
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        instance_id = data.get('instance_id', '')
                        json_instance_ids.add(instance_id)
                        
                        if instance_id not in sql_files:
                            missing_ids.append(instance_id)
                    except:
                        pass
    
    print(f"JSON中的总instance_id数: {len(json_instance_ids)}")
    print(f"SQL文件夹中的文件数: {len(sql_files)}")
    print(f"缺失的instance_id数: {len(missing_ids)}")
    
    if missing_ids:
        print(f"前10个缺失的instance_id示例: {missing_ids[:10]}")