import json
import glob
import pandas as pd


files = glob.glob("../evolutionary_history/prediction_with_avg_time/3_ep_dur_mul_0_predictive_evolution/*.json")

data = []

for file in files:
    with open(file, "r") as f:
        content = json.load(f)
        data.extend(content)

# convert to DataFrame
df = pd.DataFrame(data)

result = (
    df.groupby("success_rate")["prediction success rate"]
    .std()
    .reset_index()
    .rename(columns={"prediction success rate": "std_prediction_success_rate"})
)

result = result.sort_values("success_rate")
print(result)