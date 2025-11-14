import gymnasium as gym
import numpy as np
import neat
import os
import pickle
import pandas as pd
import time
import json
import glob

# ation space: Discrete(4)
# 0 -> do nothing
# 1 -> fire left orientation engine
# 2 -> fire main engine
# 3 -> fire right orientation engine


# observation space:
# 0 -> x position
# 1 -> y position
# 2 -> x velocity
# 3 -> y velocity
# 4 -> lander angle
# 5 -> angular velocity
# 6 -> left leg contact
# 7 -> right leg contact

# Reward structure:
# increase/decrease based on the distance to landing pad
# increase/decrease based on the speed
# decreased the more the lander is tilted
# -0.3 main engine firing
# -0.03 side engine firing
# +100 landing
# -100 crash
# +10 for leg contact
# + 200 for solved
STEP_LIMIT = 500
MASTER_CSV = "dataset.csv"

current_generation = 0
evo = 0


class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation


# --- Evaluate one genome ---
def eval_genome(genome, config, n_episodes, data_path="data"):
    """Evaluate a genome and log one CSV row per simulation, keeping the full time series."""
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    os.makedirs(data_path, exist_ok=True)

    all_episodes, success, duration = [], [], []

    all_rewards = []

    for episode_id in range(n_episodes):
        obs, _ = env.reset(seed=None)
        total_reward = 0
        rewards, observations, actions =  [], [], []

        for _ in range(STEP_LIMIT):
            observations.append(obs.tolist())
            action_values = net.activate(obs)
            action = np.clip(action_values, -1.0, 1.0)
            actions.append(action.tolist())

            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            total_reward += reward

            if terminated or truncated:
                break

        all_rewards.append(total_reward)

        duration.append(len(rewards) / env.metadata.get("render_fps"))

        # success heuristic (can be improved)
        if total_reward >= 200:
            success.append(1)
        else:
            success.append(0)

        # Serialize to JSON strings so each episode fits in one CSV cell
        episode_data = {
            "evolution": evo,
            "generation": current_generation,
            "genome_id": getattr(genome, "key", None),
            "episode_id": episode_id,
            "num_steps": len(rewards),
            "avg_duration": 0,
            "success_rate": 0,
            "observations": json.dumps(observations),
            "actions": json.dumps(actions),
        }

        all_episodes.append(episode_data)

        # ---- Compute averages ----
    avg_duration = np.mean(duration)
    success_rate = np.sum(success)/n_episodes

    # Add averages as *repeated metadata* (same value on every row)
    df = pd.DataFrame(all_episodes)
    df["avg_duration"] = avg_duration
    df["success_rate"] = success_rate

    csv_path = os.path.join(data_path, MASTER_CSV)
    df.to_csv(csv_path, mode="a", header=False, index=False)
    #print(f"🧾 Logged {len(df)} episodes for genome {getattr(genome, 'key', 'unknown')} → {csv_path}")

    env.close()
    return np.mean(all_rewards)

# --- Evaluate an entire population ---
def eval_population(genomes, config, data_path, n_episodes):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, n_episodes, data_path)

# --- Main ---
def run_neat(config_file, model_path="models/temp.pkl", data_path="data", generations=100, n_episodes=10):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    population = neat.Population(config)

    # Add some reporting (to console and stats file)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(GenerationTracker())

    # Run for up to 100 generations
    winner = population.run(lambda genomes, config: eval_population(genomes, config, data_path, n_episodes), generations)

    print("\nBest genome:\n", winner)

    # Save the winning genome
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)

    print(f"Best genome saved to {model_path}")

    # Test the best network visually
    env = gym.make("LunarLander-v3",continuous=True,enable_wind=True,wind_power=15.0,turbulence_power=1.5, render_mode="human")
    observation, _ = env.reset()
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    total_reward = 0
    for _ in range(500):
        action = np.clip(net.activate(observation), -1.0, 1.0)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print("Final test reward:", total_reward)

    env.close()


def merge_csv_files(data_path="data", output_file="all_episodes.csv"):
    # Get all CSV files in that folder
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    # Load and merge them all
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Save combined dataset
    merged_path = os.path.join(data_path, output_file)
    df.to_csv(merged_path, index=False)

    print(f"Merged {len(csv_files)} CSV files into: {merged_path}")
    print(f"Total rows: {len(df)}")

def delete_temp_csv_files(data_path="data"):
    csv_files = glob.glob(os.path.join(data_path, "evo_*.csv"))
    for f in csv_files:
        os.remove(f)
    print(f"Deleted {len(csv_files)} temporary CSV files from {data_path}")    

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    data_path = os.path.join(local_dir, "data")
    master_csv = os.path.join(data_path, MASTER_CSV)

    if not os.path.exists(master_csv):
        pd.DataFrame(columns=[
            "evolution", "generation", "genome_id", "episode_id", "num_steps",
            "avg_duration", "success_rate", "observations", "actions"
        ]).to_csv(master_csv, index=False)

    for evo in range(1):
        generations = 1
        episodes_per_genome = 10

        model_path = os.path.join(local_dir, "models", f"evolution_{evo}.pkl")

        run_neat(config_path, model_path, data_path, generations, episodes_per_genome)