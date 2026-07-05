import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import json
import h5py
import gymnasium_robotics

H5_PATH = "/mnt/DATA/miguelfazenda/pickplace/dataset/raw/train_episodes_hf_6.h5"

gym.register_envs(gymnasium_robotics)

ENV_ID        = "FetchPushDense-v4"
N_EPISODES    = 10       # evaluations per candidate


def evaluate(model, id, env, range_vals, min_vals, n_episodes=N_EPISODES):
    successes = 0
    total_rewards = []
    final_rewards = []
    durations = []

    genome_data = {
        "genome_id": id,
        "episodes_data": [],
        "success_rate": 0,
        "avg_duration": 0,
        "fitness": 0
    }

    for _ in range(n_episodes):
        episode_data = {
            "observations": [],
            "actions": [],
            "success": 0,
            "last_reward": 0,
            "duration": 0
        }
        observations, actions = [], []
        obs_dict, _ = env.reset()
        obs = np.concatenate([
            obs_dict["observation"],
            obs_dict["desired_goal"] - obs_dict["achieved_goal"]])
        done = False

        total_reward = 0
        success_step = None

        for step in range(env._max_episode_steps):
            obs = normalize_observation(obs, range_vals, min_vals)

            observations.append(obs)
            action, _states = model.predict(obs_dict, deterministic=True)
            actions.append(action)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = np.concatenate([
                obs_dict["observation"],
                obs_dict["desired_goal"] - obs_dict["achieved_goal"]])

            total_reward += reward

            if reward >= -0.05 and success_step is None:
                success_step = step
            elif reward < -0.05:
                success_step = None

            if terminated or truncated:
                break

        if reward > -0.05:  
            successes += 1
            success = 1
        else:
            success = 0

        if success_step is None:
            success_step = env._max_episode_steps - 1

        total_rewards.append(total_reward)
        final_rewards.append(reward)

        duration = success_step / env.metadata["render_fps"]
        durations.append(duration)

        episode_data["observations"] = observations
        episode_data["actions"] = actions
        episode_data["success"] = success
        episode_data["duration"] = duration
        episode_data["last_reward"] = reward
        genome_data["episodes_data"].append(episode_data)
    
    success_rate = successes / n_episodes
    fitness = np.mean(final_rewards) * 100 + success_rate * 100  # higher success and lower distance → higher fitness

    genome_data["success_rate"] = success_rate
    genome_data["avg_duration"] = float(np.mean(durations))
    genome_data["fitness"] = float(fitness)

    env.close()

    return fitness, success_rate, genome_data

def load_limits(path="observation_limits_push.json"):
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

def save_population_data_to_h5(evo,current_generation,population_data):
    with h5py.File(H5_PATH, "a") as h5file:
        for genome_data in population_data:
            """
            random_number = np.random.rand()  # Generate a random number for uniqueness
            if genome_data["success_rate"] < 0.1 and random_number > 0.005:
                continue
            elif  genome_data["success_rate"] < 0.2 and random_number > 0.005:
                continue
            elif  genome_data["success_rate"] < 0.3 and random_number > 0.01:
                continue
            elif genome_data["success_rate"] < 0.4 and random_number > 0.1:
                continue
            elif genome_data["success_rate"] < 0.5 and random_number > 0.5:
                continue
            elif genome_data["success_rate"] > 0.95:
                continue
            else:
                """ 
            group_name = f"evo_{evo}/gen_{current_generation}/genome_{genome_data['genome_id']}"
            print(f"Saving genome data to group: {group_name} with success rate: {genome_data['success_rate']}")
            if group_name in h5file:
                del h5file[group_name]  # Remove existing group to avoid duplication
            group = h5file.create_group(group_name)
            group.attrs["success_rate"] = genome_data["success_rate"]
            group.attrs["avg_duration"] = genome_data["avg_duration"]
            group.attrs["fitness"] = genome_data["fitness"]

            for episode_id, episode_data in enumerate(genome_data["episodes_data"]):
                episode_group = group.create_group(f"episode_{episode_id}")
                episode_group.create_dataset("observations", data=np.array(episode_data["observations"]),compression="gzip",
                compression_opts=4)
                episode_group.create_dataset("actions", data=np.array(episode_data["actions"]),compression="gzip",
                compression_opts=4)
                episode_group.attrs["success"] = episode_data["success"]
                episode_group.attrs["duration"] = episode_data["duration"]
                episode_group.attrs["last_reward"] = episode_data["last_reward"]


if __name__ == "__main__":
    # Create the environment
    env = gym.make(ENV_ID, max_episode_steps=150)

    # Load the model
    model = TQC.load("checkpoints/fetchpush_tqc_500000_steps.zip", env=env)

    range_vals, min_vals = load_limits()

    for i in range(25):  # Evaluate 5 different evolutions
        population_data = []
        for j in range(250):  # Evaluate 5 different genomes
            fitness, success_rate, genome_data = evaluate(model, j, env, range_vals, min_vals, n_episodes=N_EPISODES)
            population_data.append(genome_data)

            print(f"Fitness: {fitness}, Success Rate: {success_rate}")
        save_population_data_to_h5(400, i, population_data)
        