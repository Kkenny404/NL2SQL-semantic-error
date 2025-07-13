import json
import os

# Paths
JSONL_PATH = 'spider2-snow/spider2-snow.jsonl'
TRUE_SQL_DIR = 'Spider/true_sql'
OUTPUT_PATH = 'spider2/spider2-snow_with_golden.jsonl'

# Load all available golden SQL filenames (without extension)
golden_ids = set()
for fname in os.listdir(TRUE_SQL_DIR):
    if fname.endswith('.sql'):
        golden_ids.add(os.path.splitext(fname)[0])

# Filter jsonl
with open(JSONL_PATH, 'r', encoding='utf-8') as fin, open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        instance_id = obj.get('instance_id')
        if instance_id and instance_id in golden_ids:
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f'Filtered file written to {OUTPUT_PATH}')
