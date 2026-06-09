import json
import plotly.graph_objects as go

# Load data
with open("trajectory_differences_1_episode_predictive.json", "r") as f:
    evo1 = json.load(f)

with open("trajectory_differences_3_episodes_predictive.json", "r") as f:
    evo2 = json.load(f)

with open("trajectory_differences_10_episodes_fitness.json", "r") as f:
    evo3 = json.load(f)

# Create histogram figure
fig = go.Figure()
"""
fig.add_trace(go.Histogram(
    x=evo1,
    name="Predictive Evolution (1 episode)",
    opacity=0.6
))
"""
fig.add_trace(go.Histogram(
    x=evo2,
    name="Predictive Evolution (3 episodes)",
    opacity=0.6
))


fig.add_trace(go.Histogram(
    x=evo3,
    name="Predictive Evolution (10 episodes)",
    opacity=0.6
))

# Layout settings
fig.update_layout(
    title="Trajectory Difference Distributions",
    xaxis_title="Trajectory Difference",
    yaxis_title="Count",
    barmode="overlay",   # overlay histograms
)

# Save to HTML
fig.write_html("trajectory_histograms_3_v_10.html")

fig.show()