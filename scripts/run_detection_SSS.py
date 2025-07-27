import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from utils import build_prompt, build_cot_prompt, parse_answer, query_gemini, query_claude, query_gpt, load_sql
from schema.get_spider_schema import get_schema, schema_shrink

# ========== 配置数据集 ==========
DATASETS = {
    "1": {
        "name": "SSS",
        "data_path": "SSS-data/random_order_ground_truth.json",
        "db_root": "spider2-lite/resource/databases",
        "sql_root": "SSS-data/sql"
    }
}

# ========== 用户交互 ==========
def choose_dataset():
    print("Please choose a dataset:")
    for k, v in DATASETS.items():
        print(f"{k}. {v['name']}")
    choice = input("Enter the number to choose: ").strip()
    return DATASETS.get(choice, DATASETS["1"])  # 默认SSS

def choose_model():
    models = {
        "1": ("Claude", query_claude),
        "2": ("Gemini", query_gemini),
        "3": ("GPT-4.1", query_gpt),
    }
    print("Please choose a model:")
    for k, v in models.items():
        print(f"{k}. {v[0]}")
    choice = input("Enter the number to choose: ").strip()
    return models.get(choice, models["3"])  # 默认 GPT-4o

def choose_prompt():
    prompts = {
        "1": ("Baseline", build_prompt),
        "2": ("CoT", build_cot_prompt),
    }
    print("Please choose a prompt type:")
    for k, v in prompts.items():
        print(f"{k}. {v[0]}")
    choice = input("Enter the number to choose: ").strip()
    return prompts.get(choice, prompts["1"])  # 默认普通Prompt

# ========== 参数 ==========
MAX_EXAMPLES = None
SCHEMA_LINE_THRESHOLD = 700
MAX_RETRIES = 5

if __name__ == "__main__":
    # Step 1: 交互选择
    dataset = choose_dataset()
    model_name, query_func = choose_model()
    prompt_name, prompt_builder = choose_prompt()

    # Step 2: 加载数据
    with open(dataset["data_path"], "r") as f:
        examples = json.load(f)
    if MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]

    # Step 3: 自动路径
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # result_dir = f"results/{dataset['name']}/method_{MAX_EXAMPLES or 'all'}"
    # os.makedirs(result_dir, exist_ok=True)
    # RESULT_PATH = os.path.join(result_dir, f"{model_name}_{prompt_name}_{timestamp}.jsonl")
    # EVAL_PATH = os.path.join(result_dir, f"{model_name}_{prompt_name}_eval_{timestamp}.json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULT_PATH = f"results/{dataset['name']}/{prompt_name}_{MAX_EXAMPLES}/{model_name}_{timestamp}.jsonl"
    EVAL_PATH = f"results/{dataset['name']}/{prompt_name}_{MAX_EXAMPLES}/{model_name}_eval_{timestamp}.json"
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    # Step 4: 初始化
    preds, labels, skip_count = [], [], 0

    with open(RESULT_PATH, "w", encoding="utf-8") as result_file:
        for idx, ex in enumerate(tqdm(examples)):
            q = ex["question"]
            sql_id = ex["sql_id"]
            db_id = ex["db"]
            label = ex["label"]

            # Schema 提取 + Shrink
            try:
                schema_full = get_schema(sql_id, db_id, dataset["db_root"])
                schema = schema_shrink(schema_full) if len(schema_full.strip().split("\n")) > SCHEMA_LINE_THRESHOLD else schema_full
            except Exception as e:
                print(f"[ERROR] Failed to extract schema for {db_id}: {e}")
                skip_count += 1
                continue
            sql = load_sql(dataset["sql_root"], sql_id)
            # Prompt
            prompt = prompt_builder(q, schema, sql)

            # 调用模型
            for attempt in range(MAX_RETRIES):
                try:
                    response = query_func(prompt)
                    break
                except Exception as e:
                    if "rate limit" in str(e).lower() or "overloaded" in str(e).lower():
                        wait_time = 2 ** attempt
                        print(f"[Retry] Waiting {wait_time}s before retrying... ({e})")
                        time.sleep(wait_time)
                    else:
                        print(f"[ERROR] {db_id}: {e}")
                        response = None
                        break

            if response is None:
                continue

            pred = parse_answer(response)
            if pred is None:
                print(f"[WARN] Could not parse LLM response for {db_id}: {response}")
                continue

            preds.append(pred)
            labels.append(label)

            result_file.write(json.dumps({
                "id": ex.get("id", idx),
                "question": q,
                "sql_id": sql_id,
                "db_id": db_id,
                "label": label,
                "prediction": pred,
                "response_raw": response,
                # "schema": schema,
                # "sql": sql,
                # "prompt": prompt
            }, ensure_ascii=False) + "\n")

    # Step 5: 评估
    if preds:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        accuracy = accuracy_score(labels, preds)
        pp = precision_score(labels, preds)
        pr = recall_score(labels, preds)
        np = tn / (tn + fn) if (tn + fn) > 0 else 0
        nr = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        tn = fp = fn = tp = accuracy = pp = pr = np = nr = 0

    evaluation_summary = {
        "dataset": dataset["name"],
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

    with open(EVAL_PATH, "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    print(f"\n[{model_name}] [{prompt_name}] Evaluation summary saved to: {EVAL_PATH}")
    print(f"Results saved to: {RESULT_PATH}")
