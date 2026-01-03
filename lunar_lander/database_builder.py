import gymnasium as gym
import numpy as np
import neat
import os
import pickle
import pandas as pd
import json
import glob
import argparse
import h5py
from datetime import datetime

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
H5_PATH = "data/episodes.h5"

current_generation = 0
evo = 0


class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation

# --- Evaluate one genome ---
def eval_genome(genome, config, n_episodes):
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    success_count = 0
    durations = []
    total_rewards = []

    base_path = (
        f"evo_{evo}/"
        f"gen_{current_generation}/"
        f"genome_{genome.key}"
    )

    with h5py.File(H5_PATH, "a") as h5:
        for episode_id in range(n_episodes):
            obs, _ = env.reset()
            observations, actions = [], []
            total_reward = 0

            for step in range(STEP_LIMIT):
                observations.append(obs)
                action = np.clip(net.activate(obs), -1.0, 1.0)
                actions.append(action)

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            duration = step / env.metadata["render_fps"]
            durations.append(duration)
            total_rewards.append(total_reward)

            if total_reward >= 200:
                success_count += 1

            ep_grp = h5.create_group(f"{base_path}/ep_{episode_id}")
            ep_grp.create_dataset(
                "observations",
                data=np.asarray(observations),
                compression="gzip",
                compression_opts=4
            )
            ep_grp.create_dataset(
                "actions",
                data=np.asarray(actions),
                compression="gzip",
                compression_opts=4
            )
            #ep_grp.attrs["reward"] = total_reward
            #ep_grp.attrs["duration"] = duration

        # ---- finalize genome-level metadata ----
        genome_grp = h5[base_path]
        genome_grp.attrs["success_rate"] = success_count / n_episodes
        genome_grp.attrs["avg_duration"] = float(np.mean(durations))

    env.close()
    return np.mean(total_rewards)

# --- Evaluate an entire population ---
def eval_population(genomes, config, n_episodes):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, n_episodes)

# --- Main ---
def run_neat(config_file, model_path="models/temp.pkl", generations=100, n_episodes=10):
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
    winner = population.run(lambda genomes, config: eval_population(genomes, config, n_episodes), generations)

    print("\nBest genome:\n", winner)

    # Save the winning genome
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)

    print(f"Best genome saved to {model_path}")

    # Test the best network visually
    """
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
    """

def parse_args():
    parser = argparse.ArgumentParser(description="Run NEAT evolutions")

    parser.add_argument(
        "--evolutions",
        type=int,
        default=1,
        help="Number of evolutions to run"
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=125,
        help="Number of generations per evolution"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per genome"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    
    n_evolutions = args.evolutions
    generations = args.generations
    episodes_per_genome = args.episodes
    datetime = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_folder = os.path.join(local_dir, "models", f"{datetime}_neat")
    os.makedirs(model_folder, exist_ok=True)

    for evo in range(n_evolutions):
        model_path = os.path.join(model_folder, f"evolution_{evo}.pkl")

        run_neat(config_path, model_path, generations, episodes_per_genome)