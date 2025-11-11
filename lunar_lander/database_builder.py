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


current_generation = 0

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

    all_episodes = []

    for episode_id in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(1_000_000))
        total_reward = 0
        rewards, observations, actions = [], [], []

        start_time = time.time()

        for _ in range(500):
            observations.append(obs.tolist())
            action_values = net.activate(obs)
            action = np.clip(action_values, -1.0, 1.0)
            actions.append(action.tolist())

            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            total_reward += reward

            if terminated or truncated:
                break

        elapsed_time = time.time() - start_time

        # success heuristic (can be improved)
        success = total_reward > 100

        # Serialize to JSON strings so each episode fits in one CSV cell
        episode_data = {
            "generation": current_generation,
            "genome_id": getattr(genome, "key", None),
            "episode_id": episode_id,
            "total_reward": total_reward,
            "num_steps": len(rewards),
            "duration_sec": elapsed_time,
            "success": success,
            "observations": json.dumps(observations),
            "actions": json.dumps(actions),
            "rewards": json.dumps(rewards),
        }

        all_episodes.append(episode_data)

    df = pd.DataFrame(all_episodes)
    csv_path = os.path.join(data_path, f"gen_{current_generation}_genome_{getattr(genome, 'key', 'unknown')}.csv")
    df.to_csv(csv_path, index=False)
    #print(f"🧾 Logged {len(df)} episodes for genome {getattr(genome, 'key', 'unknown')} → {csv_path}")

    env.close()
    return df["total_reward"].mean()
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
    csv_files = glob.glob(os.path.join(data_path, "gen_*.csv"))
    for f in csv_files:
        os.remove(f)
    print(f"Deleted {len(csv_files)} temporary CSV files from {data_path}")    

if __name__ == "__main__":
    evo = 1

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    model_path = os.path.join(local_dir, "models", f"evolution_{evo}.pkl")
    data_path = os.path.join(local_dir, "data")

    run_neat(config_path, model_path, data_path, 1, 10)

    merge_csv_files(data_path)

    delete_temp_csv_files(data_path)