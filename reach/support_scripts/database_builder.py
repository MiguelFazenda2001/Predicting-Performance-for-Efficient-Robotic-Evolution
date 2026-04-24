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
import gymnasium_robotics
from multiprocessing import Pool, cpu_count


STEP_LIMIT = 150
H5_PATH = "/mnt/DATA/miguelfazenda/pickplace/dataset/raw/test_episodes.h5"

current_generation = 0
evo = 0


class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation

# --- Evaluate one genome ---
def eval_genome(genome, config, n_episodes):
    env = gym.make('FetchReachDense-v4', max_episode_steps=STEP_LIMIT)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    success_count = 0
    durations = []
    total_rewards = []
    final_rewards = []

    genome_data = {
        "genome_id": genome.key,
        "episodes_data": [],
        "success_rate": 0,
        "avg_duration": 0,
    }

    for episode_id in range(n_episodes):
        episode_data = {
            "observations": [],
            "actions": [],
            "reward": 0,
            "duration": 0,
        }
        dict_obs, _ = env.reset()
        observations, actions = [], []
        total_reward = 0

        for step in range(STEP_LIMIT):
            obs = np.concatenate([dict_obs["observation"],dict_obs["desired_goal"], dict_obs["achieved_goal"]])

            observations.append(obs)
            action = np.clip(net.activate(obs), -1.0, 1.0)
            actions.append(action)

            dict_obs, reward, terminated, truncated, info = env.step(action)

            achieved = dict_obs["achieved_goal"]
            desired = dict_obs["desired_goal"]

            distance = np.linalg.norm(achieved - desired)

            if step == 0:
                prev_distance = distance

            progress = prev_distance - distance
            prev_distance = distance

            total_reward += reward + 2.0 * progress

            if terminated or truncated:
                break

        if reward >= -0.05:
            total_reward += 5000.0
            success = 1
            success_count += 1
            print(f"Genome {genome.key} Success in episode {episode_id} with reward {reward:.2f} and total_reward {total_reward:.2f}")
        else:
            success = 0

        final_rewards.append(reward)

        duration = step / env.metadata["render_fps"]
        durations.append(duration)
        total_rewards.append(total_reward)

        episode_data["observations"] = observations
        episode_data["actions"] = actions
        episode_data["reward"] = reward
        episode_data["success"] = success
        episode_data["duration"] = duration
        genome_data["episodes_data"].append(episode_data)

    genome_data["success_rate"] = success_count / n_episodes
    genome_data["avg_duration"] = float(np.mean(durations))

    env.close()
    return np.mean(total_rewards), genome_data #np.mean(total_rewards)

# --- Evaluate an entire population ---
def eval_population(genomes, config, n_episodes):
    population_data = []
    
    genome_map = dict(genomes)

    args = [(genome, config, n_episodes) for _, genome in genomes]

    with Pool(processes=25) as pool:
        results = pool.map(eval_worker, args)

    for genome_id, reward, genome_data in results:        
        genome_map[genome_id].fitness = reward
        population_data.append(genome_data)
    
    save_population_data_to_h5(population_data)

def eval_worker(args):
    genome, config, n_episodes = args
    reward, genome_data = eval_genome(genome, config, n_episodes)
    return genome.key, reward, genome_data

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
        default=200,
        help="Number of generations per evolution"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per genome"
    )

    return parser.parse_args()

def save_population_data_to_h5(population_data):
    with h5py.File(H5_PATH, "a") as h5file:
        for genome_data in population_data:
            group_name = f"evo_{evo}/gen_{current_generation}/genome_{genome_data['genome_id']}"
            if group_name in h5file:
                del h5file[group_name]  # Remove existing group to avoid duplication
            group = h5file.create_group(group_name)
            group.attrs["success_rate"] = genome_data["success_rate"]
            group.attrs["avg_duration"] = genome_data["avg_duration"]

            for episode_id, episode_data in enumerate(genome_data["episodes_data"]):
                episode_group = group.create_group(f"episode_{episode_id}")
                episode_group.create_dataset("observations", data=np.array(episode_data["observations"]),compression="gzip",
                compression_opts=4)
                episode_group.create_dataset("actions", data=np.array(episode_data["actions"]),compression="gzip",
                compression_opts=4)
                episode_group.attrs["reward"] = episode_data["reward"]
                episode_group.attrs["success"] = episode_data["success"]
                episode_group.attrs["duration"] = episode_data["duration"]

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

    print("Starting NEAT evolutions...")

    for evo in range(n_evolutions):
        model_path = os.path.join(model_folder, f"evolution_{evo}.pkl")

        run_neat(config_path, model_path, generations, episodes_per_genome)