# enhanced_main.py - 基于你的原始代码，只修改关键部分
import json
import os
import time
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import extract_schema_from_sqlite, query_gpt, query_gemini, query_claude  # 保持你原有的utils
from duck_reflex import SemanticErrorDetector  # 新增的语义检测器

# 在文件开头添加
print("=== Script Starting ===")
print(f"Current working directory: {os.getcwd()}")
# 配置参数
DATA_PATH = "bug-data/NL2SQL-Bugs-Subset.json"
DB_ROOT = "BIRD/dev_20240627/dev_databases"
MAX_EXAMPLES = None  # None for all

# 选择检测方法: 'baseline', 'rubber_duck', 'adaptive'
DETECTION_METHOD = 'adaptive'  # 推荐使用adaptive

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = f"results/Duck_reflex/Duck_reflex_Claude_{DETECTION_METHOD}_{MAX_EXAMPLES}_{timestamp}.jsonl"

# 加载数据
with open(DATA_PATH, "r") as f:
    examples = json.load(f)

if MAX_EXAMPLES is not None:
    examples = examples[:MAX_EXAMPLES]

# 初始化语义检测器
detector = SemanticErrorDetector(query_claude)

preds = []
labels = []
skip_count = 0

os.makedirs("results/Duck_reflex", exist_ok=True)
result_file = open(RESULT_PATH, "w")

MAX_RETRIES = 5

for idx, ex in enumerate(tqdm(examples)):
    q = ex["question"]
    sql = ex["sql"]
    db_id = ex["db_id"]
    label = ex["label"]  # True if correct

    db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        print(f"[SKIP] Missing DB file for {db_id}")
        skip_count += 1
        continue

    try:
        schema = extract_schema_from_sqlite(db_path)
        
        # 根据选择的方法进行检测
        if DETECTION_METHOD == 'baseline':
            # 使用你原来的简单提示词
            from utils import build_prompt, parse_answer
            prompt = build_prompt(q, schema, sql)
            
            # 重试逻辑（保持你原有的）
            for attempt in range(MAX_RETRIES):
                try:
                    response = query_gpt(prompt)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "overloaded" in error_str:
                        wait_time = 2 ** attempt
                        print(f"[Retry] Waiting {wait_time}s before retrying... ({e})")
                        time.sleep(wait_time)
                    else:
                        raise e
            else:
                print(f"[ERROR] Max retries exceeded for {db_id}")
                continue
            
            pred = parse_answer(response)
            detection_details = {
                'method_used': 'baseline',
                'response': response,
                'final_prediction': pred
            }
            
        else:
            # 使用增强的语义检测器
            detection_result = detector.detect(q, schema, sql)
            pred = detection_result['final_prediction']
            detection_details = detection_result

        if pred is not None:
            preds.append(pred)
            labels.append(label)

            # 保存详细结果
            result_data = {
                "id": ex.get("id", idx),
                "question": q,
                "sql": sql,
                "db_id": db_id,
                "label": label,
                "prediction": pred,
                "schema": schema,
                "detection_method": DETECTION_METHOD,
                "response_raw": detection_details
            }

            
            result_file.write(json.dumps(result_data, ensure_ascii=False) + "\n")

        else:
            print(f"[WARN] Could not parse LLM response for {db_id}")

    except Exception as e:
        print(f"[ERROR] {db_id}: {e}")
        continue

result_file.close()

# =====================================
# ========== Evaluation ===============
# =====================================

if len(preds) > 0 and len(labels) > 0:
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    # 计算指标
    accuracy = accuracy_score(labels, preds)
    pp = precision_score(labels, preds)  # Positive Precision
    pr = recall_score(labels, preds)     # Positive Recall
    np = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Precision
    nr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Negative Recall
    
    # 保存评估结果
    evaluation_summary = {
        "result_path": RESULT_PATH,
        "detection_method": DETECTION_METHOD,
        "total_examples": len(examples),
        "processed": len(preds),
        "skipped": skip_count,
        "accuracy": round(accuracy, 4),
        "precision": round(pp, 4),
        "recall": round(pr, 4),
        "negative_precision": round(np, 4),
        "negative_recall": round(nr, 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    
    eval_path = f"results/Duck_reflex/evaluation_Claude_{DETECTION_METHOD}_{timestamp}.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # 打印结果
    print("\n===== Enhanced Semantic Detection Results =====")
    print(f"Detection Method: {DETECTION_METHOD}")
    print(f"Result saved to: {RESULT_PATH}")
    print(f"Total Processed: {len(preds)}/{len(examples)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {pp:.4f}")
    print(f"Recall: {pr:.4f}")
    print(f"Negative Precision: {np:.4f}")
    print(f"Negative Recall: {nr:.4f}")
    print(f"Evaluation summary: {eval_path}")

else:
    print("[ERROR] No valid predictions generated!")