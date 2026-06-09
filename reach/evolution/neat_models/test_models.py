import pickle
import os
import neat
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import json

def test_saved_model(range_vals, min_vals, seed, config_file, model_path="best_genome_predictive_evolution.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load config and genome
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    with open(model_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = gym.make('FetchReachDense-v4', max_episode_steps=150)
    dict_obs, _ = env.reset(seed=seed)
    total_reward = 0
    trajectory = []

    for _ in range(150):
        obs = np.concatenate([dict_obs["observation"],dict_obs["desired_goal"] - dict_obs["achieved_goal"]])
        obs = normalize_observation(obs, min_vals, range_vals)

        action = np.clip(net.activate(obs), -1.0, 1.0)
        dict_obs, reward, terminated, truncated, _ = env.step(action)
        
        step = np.concatenate((obs, action))
        trajectory.append(step)

        total_reward += reward
        if terminated or truncated:
            break

    print("🎯 Test run total reward:", total_reward)
    env.close()

    return trajectory

def calculate_trajectory_differences(trajectories):
    differences = []

    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            traj_i = np.array(trajectories[i])
            traj_j = np.array(trajectories[j])

            max_length = max(len(traj_i), len(traj_j))
            feature_dim = traj_i.shape[1]

            padded_i = np.zeros((max_length, feature_dim))
            padded_j = np.zeros((max_length, feature_dim))

            padded_i[:len(traj_i)] = traj_i
            padded_j[:len(traj_j)] = traj_j

            diff = np.linalg.norm(padded_i - padded_j, axis=1)
            mean_diff = np.mean(diff)

            differences.append(float(mean_diff))

    return differences

def normalize_observation(obs, min_vals, range_vals):
    # scale to [0,1]
    norm = (obs - min_vals) / range_vals

    # scale to [-1,1]
    norm = norm * 2.0 - 1.0

    return norm

def load_limits(path="observation_limits_13_obs.json"):
    with open(path, "r") as f:
        data = json.load(f)

    min_vals = np.array(data["min"])
    max_vals = np.array(data["max"])

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8

    return range_vals, min_vals


if __name__ == "__main__":
    evo = 0

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    seed = [8186962, 6069524, 4234384, 8263567, 7736481, 5384850, 655040, 7454027, 5607204, 775357, 3673588, 6845558, 1199703, 6604421, 1220700, 7678301, 223337, 1956206, 8700902, 5703555, 4048572, 1184970, 2520776, 4810898, 6419877]

    neat_model_dir = "30_total_episodes_3_episodes_predictive_evolution_300_gens/"

    all_trajectories_differences = []

    range_vals, min_vals = load_limits()

    for seed_value in seed:
        print(f"Testing with seed: {seed_value}")
        all_trajectories = []

        for file in os.listdir(neat_model_dir):
            if file.endswith(".pkl"):
                print(f"Testing model: {file}")
                model_path = os.path.join(neat_model_dir, file)

                trajectory = test_saved_model(range_vals=range_vals, min_vals=min_vals, seed=seed_value, config_file=config_path, model_path=model_path)
                all_trajectories.append(trajectory)

        trajectories_differences = calculate_trajectory_differences(all_trajectories)
        all_trajectories_differences.extend(trajectories_differences)

    with open("trajectory_differences_3_episodes_predictive.json", "w") as f:
        json.dump(all_trajectories_differences, f)