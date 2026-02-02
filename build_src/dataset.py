from datasets import load_dataset, concatenate_datasets
import json

# Math dataset
ds = load_dataset("hon9kon9ize/infinite-rl")
df = ds["train"].to_pandas()
df_math = df[df["task"] == "math"]
df_math["rating"] = 0
records = df_math.to_dict(orient="records")

with open("assets/math.json", "w") as f:
    json.dump(records, f)

print(f"Successfully wrote {len(records)} math problems to assets/math.json")

# Truthy dataset
ds_yue = load_dataset("hon9kon9ize/yue-truthy", "yue")["train"]
ds_zh = load_dataset("hon9kon9ize/yue-truthy", "zh")["train"]
ds_en = load_dataset("jondurbin/truthy-dpo-v0.1")["train"]
ds_truthy = concatenate_datasets([ds_yue, ds_zh, ds_en])
df_truthy = ds_truthy.to_pandas()
records = df_truthy.to_dict(orient="records")

with open("assets/truthy.json", "w") as f:
    json.dump(records, f)

print(f"Successfully wrote {len(records)} truthy examples to assets/truthy.json")
