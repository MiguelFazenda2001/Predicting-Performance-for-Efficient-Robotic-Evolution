import gymnasium as gym
import numpy as np
import json
import gymnasium_robotics

def explore_observation_space(env, episodes=50, steps_per_episode=300):
    min_vals = None
    max_vals = None

    all_obs = []

    for ep in range(episodes):
        dict_obs, _ = env.reset()
        print(f"Episode {ep+1}/{episodes} - Starting exploration...")

        for step in range(steps_per_episode):
            #print(f"  Step {step+1}/{steps_per_episode}", end="\r")
            obs = np.concatenate([
                dict_obs["observation"],
                dict_obs["desired_goal"] - dict_obs["achieved_goal"]
            ])

            if min_vals is None:
                min_vals = np.copy(obs)
                max_vals = np.copy(obs)
            else:
                min_vals = np.minimum(min_vals, obs)
                max_vals = np.maximum(max_vals, obs)

            all_obs.append(obs)

            # random action to explore
            action = env.action_space.sample()
            dict_obs, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

    all_obs = np.array(all_obs)

    # Detect fixed (non-changing) dimensions
    std_vals = np.std(all_obs, axis=0)
    fixed_dims = np.where(std_vals < 1e-6)[0]

    results = {
        "min": min_vals.tolist(),
        "max": max_vals.tolist(),
        "fixed_dimensions": fixed_dims.tolist(),
        "std": std_vals.tolist()
    }

    return results


if __name__ == "__main__":
    env = gym.make('FetchPickAndPlaceDense-v4', max_episode_steps=500)

    results = explore_observation_space(env, episodes=25000, steps_per_episode=200)

    # Print summary
    print("\n--- Observation Space Analysis ---")
    for i, (mn, mx) in enumerate(zip(results["min"], results["max"])):
        print(f"Dim {i}: min={mn:.4f}, max={mx:.4f}")

    print("\nFixed dimensions (likely constants like goal positions):")
    print(results["fixed_dimensions"])

    # Save to file
    with open("observation_limits_13_obs_2.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved to observation_limits.json")