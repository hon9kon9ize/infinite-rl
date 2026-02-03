from datasets import load_dataset, concatenate_datasets
import pandas as pd
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
df_yue = ds_yue.to_pandas()
df_yue["lang"] = "yue"
df_zh = ds_zh.to_pandas()
df_zh["lang"] = "zh"
df_en = ds_en.to_pandas()
df_en["lang"] = "en"
df_truthy = pd.concat([df_yue, df_zh, df_en], ignore_index=True)
records = df_truthy.to_dict(orient="records")

with open("assets/truthy.json", "w") as f:
    json.dump(records, f)

print(f"Successfully wrote {len(records)} truthy examples to assets/truthy.json")
