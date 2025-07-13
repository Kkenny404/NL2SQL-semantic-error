import json
import os
import time
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import extract_schema_from_sqlite, build_prompt, parse_answer, query_gemini, query_claude, build_cot_prompt

# path
DATA_PATH = "bug-data/NL2SQL-Bugs-Subset.json"
DB_ROOT = "BIRD/dev_20240627/dev_databases"
MAX_EXAMPLES = None # None for all, and running is super slow for whole dataset, so use a small number for now

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = f"results/CoT_Gemini_results_{MAX_EXAMPLES}_{timestamp}.jsonl"

# load data
with open(DATA_PATH, "r") as f:
    examples = json.load(f)

if MAX_EXAMPLES is not None:
    examples = examples[:MAX_EXAMPLES]

preds = []
labels = []
skip_count = 0

os.makedirs("results", exist_ok=True)
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
        prompt = build_cot_prompt(q, schema, sql) 
        # retry logic
        for attempt in range(MAX_RETRIES):
            try:
                response = query_gemini(prompt)
                break
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "overloaded" in error_str:
                    wait_time = 2 ** attempt
                    print(f"[Retry] Waiting {wait_time}s before retrying... ({e})")
                    time.sleep(wait_time)
                else:
                    raise e
        else:  # If we exhausted all retries
            print(f"[ERROR] Max retries exceeded for {db_id}")
            continue

        pred = parse_answer(response)

        if pred is not None:
            preds.append(pred)
            labels.append(label)

            result_file.write(json.dumps({
                "id": ex.get("id", idx),  # 用原始文件的id，如果没有就用idx
                "question": q,
                "sql": sql,
                "db_id": db_id,
                "label": label,
                "prediction": pred,
                "response_raw": response,
                "schema": schema,
                "prompt": prompt
            }, ensure_ascii=False) + "\n")

        else:
            print(f"[WARN] Could not parse LLM response: {response}")

    except Exception as e:
        print(f"[ERROR] {db_id}: {e}")
        continue

result_file.close()


# =====================================
# ========== Evaluation ===============
# =====================================

# Check if we have any valid predictions
if len(labels) == 0 or len(preds) == 0:
    print(f"\n[ERROR] No valid predictions found. All {len(examples)} examples were skipped or failed.")
    print("Please check:")
    print("1. Database files exist in the specified path")
    print("2. LLM responses are being parsed correctly")
    print("3. API credentials are working")
    exit(1)

print(f"\nProcessed {len(labels)} valid examples out of {len(examples)} total examples.")
print(f"Unique labels: {set(labels)}")
print(f"Unique predictions: {set(preds)}")

# Get confusion matrix: tn, fp, fn, tp
try:
    cm = confusion_matrix(labels, preds, labels=[False, True])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle case where only one class is present
        print(f"[WARN] Confusion matrix has unexpected shape: {cm.shape}")
        unique_labels = set(labels) | set(preds)
        if len(unique_labels) == 1:
            label_value = list(unique_labels)[0]
            if label_value == True:
                tn, fp, fn, tp = 0, 0, 0, len(labels)
            else:
                tn, fp, fn, tp = len(labels), 0, 0, 0
        else:
            # Fallback: manual calculation
            tn = fp = fn = tp = 0
            for true_label, pred_label in zip(labels, preds):
                if true_label == True and pred_label == True:
                    tp += 1
                elif true_label == False and pred_label == False:
                    tn += 1
                elif true_label == False and pred_label == True:
                    fp += 1
                elif true_label == True and pred_label == False:
                    fn += 1
except Exception as e:
    print(f"[ERROR] Failed to compute confusion matrix: {e}")
    tn = fp = fn = tp = 0

# Accuracy
try:
    accuracy = accuracy_score(labels, preds)
except:
    accuracy = 0.0

# Positive class metrics
try:
    pp = precision_score(labels, preds, zero_division=0)  # Positive Precision
    pr = recall_score(labels, preds, zero_division=0)     # Positive Recall
except:
    pp = pr = 0.0

# Negative class metrics
np = tn / (tn + fn) if (tn + fn) > 0 else 0
nr = tn / (tn + fp) if (tn + fp) > 0 else 0

# Save evaluation result to file
evaluation_summary = {
    "result_path": RESULT_PATH,
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

eval_path = f"results/eval_CLAUDE_summary_{MAX_EXAMPLES}_{timestamp}.json"
with open(eval_path, "w") as f:
    json.dump(evaluation_summary, f, indent=2)
print(f"\nEvaluation summary saved to: {eval_path}")

# terminal
print("\n===== Evaluation Result =====")
print(f"Result saved to: {RESULT_PATH}")
print(f"Total Skipped: {skip_count}")
print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
print(f"Precision: {precision_score(labels, preds):.4f}")
print(f"Recall: {recall_score(labels, preds):.4f}")
print(f"Positive Precision (PP): {pp:.4f}")
print(f"Positive Recall (PR): {pr:.4f}")
print(f"Negative Precision (NP): {np:.4f}")
print(f"Negative Recall (NR): {nr:.4f}")