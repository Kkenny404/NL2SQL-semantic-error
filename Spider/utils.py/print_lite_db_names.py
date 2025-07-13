import json

# 输入文件路径
LITE_JSONL = 'Spider/spider2-lite_with_golden.jsonl'

bq_dbs = set()
sf_dbs = set()
local_dbs = set()

with open(LITE_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        instance_id = obj.get('instance_id', '')
        db_name = obj.get('db', '')
        if instance_id.startswith('bq'):
            bq_dbs.add(db_name)
        elif instance_id.startswith('sf'):
            sf_dbs.add(db_name)
        elif instance_id.startswith('local'):
            local_dbs.add(db_name)


# 保存为json文件
result = {
    'bq': sorted(bq_dbs),
    'sf': sorted(sf_dbs),
    'local': sorted(local_dbs)
}
with open('Spider/lite_db_names.json', 'w', encoding='utf-8') as fout:
    json.dump(result, fout, ensure_ascii=False, indent=2)
print('Saved to Spider/lite_db_names.json')
