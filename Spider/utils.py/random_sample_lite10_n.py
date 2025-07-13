import json
import random

import sys

# 用户参数
ERROR_TYPE = 'Attribute-related Errors'  # 主错误类型
N = 3  # 重复次数
SAMPLE_SIZE = 10
SUB_ERROR_TYPES = ["Attribute Mismatch", "Attribute Redundancy", "Attribute Missing"]  # 长度应等于N

INPUT_PATH = 'Spider/spider2-lite_with_golden.jsonl'
OUTPUT_PATH = f'Spider/9Errors/spider2-lite_{ERROR_TYPE}_errors.jsonl'

assert len(SUB_ERROR_TYPES) == N, "sub_error_types array length must match N"

def inject_error_fields(obj, error_type, sub_error_type):
    obj['label'] = False
    obj['error_types'] = [
        {
            'error_type': error_type,
            'sub_error_type': [sub_error_type]
        }
    ]
    return obj

# 读取所有条目
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    lines = [line for line in f if line.strip()]

results = []
for i in range(N):
    sample = random.sample(lines, SAMPLE_SIZE)
    for line in sample:
        obj = json.loads(line)
        obj = inject_error_fields(obj, ERROR_TYPE, SUB_ERROR_TYPES[i])
        results.append(obj)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
    for obj in results:
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f'Wrote {N} random samples of {SAMPLE_SIZE} to {OUTPUT_PATH}')
