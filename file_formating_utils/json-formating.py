import json

with open("Spider/error_explaination.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("Spider/error_explaination.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)