import json
import glob
import pandas as pd
import plotly.express as px

def load_data(file_paths, label):
    rows = []
    
    for i, file in enumerate(file_paths):
        with open(file, 'r') as f:
            data = json.load(f)
            
            for entry in data:
                rows.append({
                    "generation": entry["generation"],
                    "fitness": entry["fitness"],
                    "success_rate": entry["success_rate"],
                    "evaluation_time": entry["evaluation_time"],
                    "avg_ep_duration_all_episodes": entry["avg_ep_duration_all_episodes"],
                    "type": label,
                    "run": i
                })
    
    return rows

_10_episodes_fitness_evolution_files = glob.glob("../evolutionary_history/prediction_with_avg_time/10_episodes_fitness_evolution/*.json")
_10_episodes_d_mul_25_files = glob.glob("../evolutionary_history/prediction_with_avg_time/10_ep_dur_mul_25_fitness_evolution/*.json")
_3_ep_duration_mul_0_predictive_evolution_files = glob.glob("../evolutionary_history/prediction_with_avg_time/3_ep_dur_mul_0_predictive_evolution/*.json")
_3_ep_noise_200_fitness_evolution_files = glob.glob("../evolutionary_history/prediction_with_avg_time/3_ep_noise_200_fitness_evolution/*.json")
_5_ep_noise_200_fitness_evolution_files = glob.glob("../evolutionary_history/prediction_with_avg_time/5_ep_noise_200_fitness_evolution/*.json")
_10_ep_noise_200_fitness_evolution_files = glob.glob("../evolutionary_history/prediction_with_avg_time/10_ep_noise_200_fitness_evolution/*.json")

df = pd.DataFrame(
    load_data(_10_episodes_fitness_evolution_files, "10_ep_fitness") +
    load_data(_10_episodes_d_mul_25_files, "10_ep_duration_multiplier_25") +
    load_data(_3_ep_duration_mul_0_predictive_evolution_files, "3_ep_duration_multiplier_0_predictive") + 
    load_data(_3_ep_noise_200_fitness_evolution_files, "3_ep_noise_200_fitness") +
    load_data(_5_ep_noise_200_fitness_evolution_files, "5_ep_noise_200_fitness") +
    load_data(_10_ep_noise_200_fitness_evolution_files, "10_ep_noise_200_fitness")
)



df_best_per_run = (
    df.groupby(["type", "run", "generation"])["fitness"]
    .max()   # use min() if lower fitness is better
    .reset_index()
)

df_best_mean = (
    df_best_per_run.groupby(["type", "generation"])["fitness"]
    .mean()
    .reset_index()
)

df_best_mean["metric"] = "Best per generation"

df_all_mean = (
    df.groupby(["type", "generation"])["fitness"]
    .mean()
    .reset_index()
)

df_all_mean["metric"] = "All genomes"

df_plot = pd.concat([df_best_mean, df_all_mean])

fig = px.line(
    df_plot,
    x="generation",
    y="fitness",
    color="type",
    line_dash="metric",  # distinguishes best vs all
    title="Fitness Evolution Comparison: Predictions with Avg Time",
    line_dash_map={
        "Best per generation": "dash",
        "All genomes": "solid"
    }
)

fig.update_traces(line=dict(width=3))
fig.update_layout(
    legend_title_text="",
    xaxis_title="Generation",
    yaxis_title="Fitness"
)

fig.show()
fig.write_html("fitness_prediction_with_avg_time.html")

"""
#--------------------------------------------------------------------
"""

df_best_per_run = (
    df.groupby(["type", "run", "generation"])["success_rate"]
    .max()   # use min() if lower fitness is better
    .reset_index()
)

df_best_mean = (
    df_best_per_run.groupby(["type", "generation"])["success_rate"]
    .mean()
    .reset_index()
)

df_best_mean["metric"] = "Best per generation"

df_all_mean = (
    df.groupby(["type", "generation"])["success_rate"]
    .mean()
    .reset_index()
)

df_all_mean["metric"] = "All genomes"

df_plot = pd.concat([df_best_mean, df_all_mean])

fig = px.line(
    df_plot,
    x="generation",
    y="success_rate",
    color="type",
    line_dash="metric",  # distinguishes best vs all
    line_dash_map={
        "Best per generation": "dash",
        "All genomes": "solid"
    },
    title="Success Rate Comparison: Predictions with Avg Time"
)

fig.update_traces(line=dict(width=3))
fig.update_layout(
    legend_title_text="",
    xaxis_title="Generation",
    yaxis_title="Success Rate"
)

fig.show()
fig.write_html("SR_predictiona_with_average_time.html")

"""
#--------------------------------------------------------------------
"""

df_time_gen = (
    df.groupby(["type", "run", "generation"])["evaluation_time"]
    .sum()
    .reset_index()
)

df_time_gen = df_time_gen.sort_values(by=["type", "run", "generation"])

df_time_gen["cumulative_time"] = (
    df_time_gen.groupby(["type", "run"])["evaluation_time"]
    .cumsum()
)

df_time_mean = (
    df_time_gen.groupby(["type", "generation"])["cumulative_time"]
    .mean()
    .reset_index()
)

fig = px.line(
    df_time_mean,
    x="generation",
    y="cumulative_time",
    color="type",
    title="Mean Cumulative Evaluation Time per Generation: Predictions with Avg Time"
)

fig.update_traces(line=dict(width=3))

fig.update_layout(
    xaxis_title="Generation",
    yaxis_title="Cumulative Time (seconds)",
    legend_title_text=""
)

fig.show()
fig.write_html("TT_predictions_with_average_time.html")


"""
---------------------------------------------------------------------
"""

df_all_mean_duration = (
    df.groupby(["type", "generation"])["avg_ep_duration_all_episodes"]
    .mean()
    .reset_index()
)

fig = px.line(
    df_all_mean_duration,
    x="generation",
    y="avg_ep_duration_all_episodes",
    color="type",
    title="Mean Episode Duration (All Genomes): Predictions with Avg Time"
)

fig.update_traces(line=dict(width=3))
fig.update_layout(
    legend_title_text="",
    xaxis_title="Generation",
    yaxis_title="Avg Episode Duration"
)

fig.show()
fig.write_html("avg_duration_all_genomes.html")


"""
---------------------------------------------------------------------
"""

df_best_sr = (
    df.loc[
        df.groupby(["type", "run", "generation"])["success_rate"].idxmax()
    ]
)

# Step 2: compute mean duration across runs
df_best_duration_mean = (
    df_best_sr.groupby(["type", "generation"])["avg_ep_duration_all_episodes"]
    .mean()
    .reset_index()
)

fig = px.line(
    df_best_duration_mean,
    x="generation",
    y="avg_ep_duration_all_episodes",
    color="type",
    title="Mean Episode Duration (Best Genome per Generation - by Success Rate)"
)

fig.update_traces(line=dict(width=3))
fig.update_layout(
    legend_title_text="",
    xaxis_title="Generation",
    yaxis_title="Avg Episode Duration"
)

fig.show()
fig.write_html("avg_duration_best_genome.html")