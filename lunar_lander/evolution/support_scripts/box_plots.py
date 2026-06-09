import json
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

_10_episodes_fitness_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_10_episodes_fitness_evolution/*.json")

_3_episodes_fitness_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_3_episodes_fitness_evolution/*.json")
_3_episodes_predictive_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_3_episodes_predictive_evolution/*.json")

_1_episodes_fitness_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_1_episodes_fitness_evolution/*.json")
_1_episodes_predictive_files = glob.glob("../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_1_episodes_predictive_evolution/*.json")


df = pd.DataFrame(
    load_data(_10_episodes_fitness_files, "Fitness Evolution (10 episodes)") +
    load_data(_3_episodes_fitness_files, "Fitness Evolution (3 episodes)") +
    load_data(_3_episodes_predictive_files, "Predictive Evolution (3 episodes)") +
    load_data(_1_episodes_fitness_files, "Fitness Evolution (1 episode)") +
    load_data(_1_episodes_predictive_files, "Predictive Evolution (1 episode)")
)

last_gen_numbers = (
    df.groupby(["type", "run"])["generation"]
    .max()
    .reset_index()
    .rename(columns={"generation": "last_generation"})
)

# 2. merge back to filter only last generation rows
df_last = df.merge(
    last_gen_numbers,
    left_on=["type", "run", "generation"],
    right_on=["type", "run", "last_generation"]
)

# 3. now take BEST genome in that last generation
df_best_last = (
    df_last
    .groupby(["type", "run"])["success_rate"]
    .max()
    .reset_index()
)

# Filter only the 10-episode fitness runs
ten_episode_results = df_best_last[
    df_best_last["type"] == "Fitness Evolution (10 episodes)"
]

one_episode_results = df_best_last[
    df_best_last["type"] == "Fitness Evolution (1 episode)"
]

three_episode_results = df_best_last[
    df_best_last["type"] == "Fitness Evolution (3 episodes)"
]

one_predictive_results = df_best_last[
    df_best_last["type"] == "Predictive Evolution (1 episode)"
]

three_predictive_results = df_best_last[
    df_best_last["type"] == "Predictive Evolution (3 episodes)"
]   

# Print success rates
print("10 Episodes Fitness Evolution Success Rates:")
print(ten_episode_results["success_rate"].tolist())

print("\n1 Episode Fitness Evolution Success Rates:")
print(one_episode_results["success_rate"].tolist()) 

print("\n3 Episodes Fitness Evolution Success Rates:")
print(three_episode_results["success_rate"].tolist())

print("\n1 Episode Predictive Evolution Success Rates:")
print(one_predictive_results["success_rate"].tolist())

print("\n3 Episodes Predictive Evolution Success Rates:")
print(three_predictive_results["success_rate"].tolist())


"""

fig = px.box(
    df_best_last,
    x="type",
    y="success_rate",
    color="type",
    points="all",  # shows individual runs
    title="Highest Success Rate in Last Generation"
)

fig.update_layout(
    xaxis_title="Method",
    yaxis_title="Best Success Rate",
    showlegend=False
)

fig.show()
fig.write_html("box_plot_successs_rate_last_generation_10_v_1.html")

"""