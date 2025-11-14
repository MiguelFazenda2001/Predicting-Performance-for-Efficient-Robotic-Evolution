import pickle
import os
import neat
import numpy as np
import gymnasium as gym

def test_saved_model(config_file, model_path="models/temp.pkl"):
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

    env = gym.make("LunarLander-v3", continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode="human")
    obs, _ = env.reset(seed=None)
    total_reward = 0
    for _ in range(500):
        action = np.clip(net.activate(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print("🎯 Test run total reward:", total_reward)
    env.close()

if __name__ == "__main__":
    evo = 0

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    model_path = os.path.join(local_dir, "models", f"evolution_{evo}.pkl")


    test_saved_model(config_path, model_path)    