import gymnasium as gym
import numpy as np
import neat
import os
import random

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


# --- Evaluate one genome ---
def eval_genome(genome, config, n_episodes=10):
    """Evaluate a single genome over multiple episodes and return the average fitness."""
    env = gym.make("LunarLander-v3",continuous=True,enable_wind=True,wind_power=15.0,turbulence_power=1.5)  # no render for speed
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    total_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(10000000))  # different seeds for variety
        total_reward = 0

        for _ in range(500):
            action_values = net.activate(obs)
            action = np.clip(action_values, -1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        total_rewards.append(total_reward)

    env.close()
    return np.mean(total_rewards)  # <-- average over all runs

# --- Evaluate an entire population ---
def eval_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

# --- Main ---
def run_neat(config_file, generations=100):
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

    # Run for up to 100 generations
    winner = population.run(eval_population, generations)

    print("\nBest genome:\n", winner)

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

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)