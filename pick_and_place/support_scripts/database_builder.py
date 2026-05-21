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
#env = gym.make('FetchPickAndPlaceDense-v4', max_episode_steps=STEP_LIMIT)

min_vals = None
range_vals = None


class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation

# --- Evaluate one genome ---
def eval_genome(genome, config, n_episodes):
    env = gym.make('FetchPickAndPlaceDense-v4', max_episode_steps=STEP_LIMIT)

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
            "total_rewards": 0,
        }
        dict_obs, _ = env.reset()
        observations, actions = [], []
        total_reward = 0

        for step in range(STEP_LIMIT):
            obs = np.concatenate([dict_obs["observation"],dict_obs["desired_goal"] - dict_obs["achieved_goal"]])
            obs = normalize_observation(obs, range_vals, min_vals)

            #observations.append(obs)
            action = np.clip(net.activate(obs), -1.0, 1.0)
            #actions.append(action)

            dict_obs, reward, terminated, truncated, info = env.step(action)

            gripper_pos = dict_obs["observation"][:3]
            object_pos = dict_obs["observation"][3:6]
            goal_pos = dict_obs["desired_goal"]

            object_to_goal = np.linalg.norm(goal_pos - object_pos)
            gripper_to_object = np.linalg.norm(gripper_pos - object_pos)

            lift_bonus = 1.0 if object_pos[2] > 0.45 else 0.0
            success_bonus = 50.0 if reward >= -0.05 else 0.0

            if lift_bonus > 0.0:
                print(f"Genome {genome.key} lifted the object in episode {episode_id} at step {step}")

            total_reward += (
                -object_to_goal * 5.0
                -gripper_to_object * 1.0
                +lift_bonus
                +success_bonus
            )



            if terminated or truncated:
                break

        if reward >= -0.05:
            success_count += 1
            #total_reward += 100
            #print(f"Genome {genome.key} Success in episode {episode_id} with reward {reward:.2f} and total_reward {total_reward:.2f}")
        else:
            success = 0

        final_rewards.append(reward)

        duration = step / env.metadata["render_fps"]
        durations.append(duration)
        total_rewards.append(total_reward)

        #episode_data["observations"] = observations
        #episode_data["actions"] = actions
        #episode_data["reward"] = reward
        #episode_data["success"] = success
        #episode_data["duration"] = duration
        #genome_data["episodes_data"].append(episode_data)

    genome_data["success_rate"] = success_count / n_episodes
    genome_data["avg_duration"] = float(np.mean(durations))
    genome_data["final_rewards"] = float(np.mean(final_rewards))
    genome_data["total_rewards"] = float(np.mean(total_rewards))

    env.close()
    return np.mean(total_rewards) * 10, genome_data #np.mean(total_rewards)

# --- Evaluate an entire population ---
def eval_population(genomes, config, n_episodes):
    population_data = []
    
    genome_map = dict(genomes)

    args = [(genome, config, n_episodes) for _, genome in genomes]

    with Pool(processes=50) as pool:
        results = pool.map(eval_worker, args)

    for genome_id, reward, genome_data in results:        
        genome_map[genome_id].fitness = reward
        population_data.append(genome_data)
    
    #save_population_data_to_h5(population_data)
    save_population_data_to_json(population_data)

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

def save_population_data_to_json(population_data):
    data = {}
    sr_0,sr_10,sr_20,sr_30,sr_40,sr_50,sr_60,sr_70,sr_80,sr_90,sr_100 = 0,0,0,0,0,0,0,0,0,0,0

    gen_key = f"gen_{current_generation}"

    if gen_key not in data:
        data[gen_key] = {}

    # Store genome data
    for genome_data in population_data:
        genome_id = str(genome_data["genome_id"])  # keep keys as strings for JSON

        data[gen_key][genome_id] = {
            "success_rate": genome_data["success_rate"],
            "avg_duration": genome_data["avg_duration"],
            "final_rewards": genome_data["final_rewards"],
            "total_rewards": genome_data["total_rewards"]
        }

        # Count success rates for summary
        sr = genome_data["success_rate"]
        if sr <= 0.05:
            sr_0 += 1
        elif sr <= 0.15:
            sr_10 += 1
        elif sr <= 0.25:
            sr_20 += 1
        elif sr <= 0.35:
            sr_30 += 1
        elif sr <= 0.45:
            sr_40 += 1
        elif sr <= 0.55:
            sr_50 += 1
        elif sr <= 0.65:
            sr_60 += 1
        elif sr <= 0.75:
            sr_70 += 1
        elif sr <= 0.85:
            sr_80 += 1
        elif sr <= 0.95:
            sr_90 += 1
        elif sr <= 1.05:
            sr_100 += 1

    print(f"Generation {current_generation} Success Rate Distribution:")
    print(f"0%: {sr_0}")
    print(f"10%: {sr_10}")
    print(f"20%: {sr_20}")
    print(f"30%: {sr_30}")
    print(f"40%: {sr_40}")
    print(f"50%: {sr_50}")
    print(f"60%: {sr_60}")
    print(f"70%: {sr_70}")
    print(f"80%: {sr_80}")
    print(f"90%: {sr_90}")
    print(f"100%: {sr_100}")

    # Save back to file
    with open("dataset.json", "a") as f:
        f.write(json.dumps(data) + "\n")

def load_limits(path="observation_limits_28_obs.json"):
    with open(path, "r") as f:
        data = json.load(f)

    min_vals = np.array(data["min"])
    max_vals = np.array(data["max"])

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8

    return range_vals, min_vals

def normalize_observation(obs, range_vals, min_vals):
    # scale to [0,1]
    norm = (obs - min_vals) / range_vals

    # scale to [-1,1]
    norm = norm * 2.0 - 1.0

    return norm

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
        default=500,
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

    range_vals, min_vals = load_limits()

    print("Starting NEAT evolutions...")

    for evo in range(n_evolutions):
        model_path = os.path.join(model_folder, f"evolution_{evo}.pkl")

        run_neat(config_path, model_path, generations, episodes_per_genome)