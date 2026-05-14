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
                    "type": label,
                    "run": i
                })
    
    return rows

_10_episodes_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/10_episodes_fitness_evolution/*.json")

_10_episodes_SR_fitness_evolution_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/10_episodes_SR_fitness_evolution/*.json")
_10_episodes_predictive_evolution_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/10_episodes_predictive_evolution/*.json")
sr_10_ep_noise_fitness_evolution_files = glob.glob("../evolutionary_history/prediction_with_avg_time/10_ep_noise_fitness_evolution/*.json")






df = pd.DataFrame(
    load_data(_10_episodes_files, "10_ep") +
    load_data(_10_episodes_SR_fitness_evolution_files, "10_ep_SR_fitness") +
    load_data(_10_episodes_predictive_evolution_files, "10_ep_predictive") +
    load_data(sr_10_ep_noise_fitness_evolution_files, "10_ep_noise_fitness")
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
    title="Fitness Evolution Comparison",
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
fig.write_html("normal_vs_noise_fitness_evolution_10_v_3_v_1.html")

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
    title="Success Rate Comparison"
)

fig.update_traces(line=dict(width=3))
fig.update_layout(
    legend_title_text="",
    xaxis_title="Generation",
    yaxis_title="Success Rate"
)

fig.show()
fig.write_html("SR_normal_vs_noise_fitness_evolution_10_v_3_v_1.html")

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
    title="Mean Cumulative Evaluation Time per Generation"
)

fig.update_traces(line=dict(width=3))

fig.update_layout(
    xaxis_title="Generation",
    yaxis_title="Cumulative Time (seconds)",
    legend_title_text=""
)

fig.show()
fig.write_html("TT_normal_vs_noise_fitness_evolution_10_v_3_v_1.html")