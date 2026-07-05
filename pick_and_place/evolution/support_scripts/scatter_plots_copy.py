import json
import glob
import pandas as pd
import plotly.express as px

def load_data(file_paths, label):
    rows = []

    for i, file in enumerate(file_paths):
        with open(file, 'r') as f:

            # Read line by line (JSONL format)
            for line in f:
                entry = json.loads(line)

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

_10_episodes_fitness_evolution_files = glob.glob("../evolutionary_history/30_total_episodes_10_episodes_fitness_evolution_300_gens/*.json")
_1_episodes_fitness_evolution_files = glob.glob("../evolutionary_history/30_total_episodes_1_episodes_fitness_evolution_300_gens/*.json")
_1_episodes_predictive_evolution_files = glob.glob("../evolutionary_history/30_total_episodes_1_episodes_predictive_evolution_300_gens/*.json")
_3_episodes_fitness_evolution_files = glob.glob("../evolutionary_history/30_total_episodes_3_episodes_fitness_evolution_300_gens/*.json")
_3_episodes_predictive_evolution_files = glob.glob("../evolutionary_history/30_total_episodes_3_episodes_predictive_evolution_300_gens/*.json")    

df = pd.DataFrame(
    load_data(_10_episodes_fitness_evolution_files, "Fitness Evolution (10 episodes)") +
    load_data(_1_episodes_fitness_evolution_files, "Fitness Evolution (1 episode)") +
    load_data(_1_episodes_predictive_evolution_files, "Predictive Evolution (1 episode)") +
    load_data(_3_episodes_fitness_evolution_files, "Fitness Evolution (3 episodes)")+ 
    load_data(_3_episodes_predictive_evolution_files, "Predictive Evolution (3 episodes)")
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

df_plot = pd.concat([df_all_mean, df_best_mean])

fig = px.line(
    df_plot,
    x="generation",
    y="fitness",
    color="type",
    line_dash="metric",  # distinguishes best vs all
    title="Fitness Values",
    line_dash_map={
        "Best per generation": "dash",
        "All genomes": "solid"
    }
)

# Make lines thicker
fig.update_traces(line=dict(width=3))

# Hide duplicate legends
seen = set()

for trace in fig.data:
    name = trace.name.split(",")[0]  # keep only the type name

    if name in seen:
        trace.showlegend = False
    else:
        trace.name = name
        seen.add(name)

# Add style explanation traces
fig.add_scatter(
    x=[None],
    y=[None],
    mode="lines",
    line=dict(color="black", dash="solid", width=3),
    name="Solid = Mean"
)

fig.add_scatter(
    x=[None],
    y=[None],
    mode="lines",
    line=dict(color="black", dash="dash", width=3),
    name="Dashed = Best"
)

fig.update_layout(
    legend_title_text="",
    xaxis_title="Generation",
    yaxis_title="Fitness"
)

fig.show()
fig.write_html("F_predictive_vs_sr_evolution_10_v_3_v_1.html")

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

df_plot = pd.concat([df_all_mean, df_best_mean])

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
    title="Success Rate"
)

# Make lines thicker
fig.update_traces(line=dict(width=3))

# Hide duplicate legends
seen = set()

for trace in fig.data:
    name = trace.name.split(",")[0]  # keep only the type name

    if name in seen:
        trace.showlegend = False
    else:
        trace.name = name
        seen.add(name)

# Add style explanation traces
fig.add_scatter(
    x=[None],
    y=[None],
    mode="lines",
    line=dict(color="black", dash="solid", width=3),
    name="Solid = Mean"
)

fig.add_scatter(
    x=[None],
    y=[None],
    mode="lines",
    line=dict(color="black", dash="dash", width=3),
    name="Dashed = Best"
)

fig.update_layout(
    legend_title_text="",
    xaxis_title="Generation",
    yaxis_title="Success Rate"
)

fig.show()
fig.write_html("SR_predictive_vs_sr_evolution_10_v_3_v_1.html")

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
    title="Cumulative Evaluation Time"
)

# Make lines thicker
fig.update_traces(line=dict(width=3))

fig.update_layout(
    xaxis_title="Generation",
    yaxis_title="Cumulative Time (seconds)",
    legend_title_text=""
)

fig.show()
fig.write_html("TT_predictive_vs_sr_evolution_10_v_3_v_1.html")