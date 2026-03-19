import json
import pandas as pd
import matplotlib.pyplot as plt

# Load files
with open("fitness_evolution_history.json") as f:
    normal_data = json.load(f)

with open("predictive_evolution_history.json") as f:
    predictive_data = json.load(f)

# Convert to DataFrames
df_normal = pd.DataFrame(normal_data)
df_pred = pd.DataFrame(predictive_data)

def aggregate(df):
    return df.groupby("generation").agg({
        "fitness": ["mean", "max"],
        "success_rate": ["mean", "max"],
        "evaluation_time": "sum"
    }).reset_index()

agg_normal = aggregate(df_normal)
agg_pred = aggregate(df_pred)

# Flatten column names
agg_normal.columns = ["generation", "fitness_mean", "fitness_max",
                      "success_mean", "success_max", "time_sum"]

agg_pred.columns = agg_normal.columns

plt.figure()

plt.plot(agg_normal["generation"], agg_normal["fitness_mean"], label="Normal Mean", color='blue')
plt.plot(agg_normal["generation"], agg_normal["fitness_max"], linestyle="--", label="Normal Best", color='blue')

plt.plot(agg_pred["generation"], agg_pred["fitness_mean"], label="Predictive Mean", color='orange')
plt.plot(agg_pred["generation"], agg_pred["fitness_max"], linestyle="--", label="Predictive Best", color='orange')

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Evolution")
plt.legend()

plt.savefig("fitness_plot.png", dpi=300)

plt.figure()

plt.plot(agg_normal["generation"], agg_normal["success_mean"], label="Normal Mean", color='blue')
plt.plot(agg_normal["generation"], agg_normal["success_max"], linestyle="--", label="Normal Best", color='blue')

plt.plot(agg_pred["generation"], agg_pred["success_mean"], label="Predictive Mean", color='orange')
plt.plot(agg_pred["generation"], agg_pred["success_max"], linestyle="--", label="Predictive Best", color='orange')

plt.xlabel("Generation")
plt.ylabel("Success Rate")
plt.title("Success Rate Evolution")
plt.legend()

plt.savefig("success_plot.png", dpi=300)

plt.figure()

plt.plot(agg_normal["generation"], agg_normal["time_sum"].cumsum(), label="Normal Cumulative Time", color='blue')
plt.plot(agg_pred["generation"], agg_pred["time_sum"].cumsum(), label="Predictive Cumulative Time", color='orange')

plt.xlabel("Generation")
plt.ylabel("Time (seconds)")
plt.title("Cumulative Evaluation Time")
plt.legend()

plt.savefig("cumulative_time_plot.png", dpi=300)