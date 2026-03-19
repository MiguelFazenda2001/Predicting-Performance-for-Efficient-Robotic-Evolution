from fitness_evolution import FitnessEvolution
from predictive_evolution import PredictiveEvolution
import json
import time
import numpy as np


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


if __name__ == "__main__":

    fitness_evolver = FitnessEvolution(100, 10, STEP_LIMIT)

    fitness_evolutions_history = fitness_evolver.evolve("config-feedforward.txt", "best_genome_fitness_evolution.pkl")

    json_path = "fitness_evolution_history.json"
    with open(json_path, "w") as f:
        json.dump(fitness_evolutions_history, f, indent=4)
    
    x_mean = np.load("x_mean.npy")
    x_std = np.load("x_std.npy")
    y_mean = np.load("y_mean.npy")
    y_std = np.load("y_std.npy")

    predictive_evolver = PredictiveEvolution(num_generations=100, n_episodes_to_predict=3, total_n_episodes=10, model_path="predictive_model_3_ep.pth", input_dim=10, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, step_limit=STEP_LIMIT)

    predictive_evolutions_history = predictive_evolver.evolve("config-feedforward.txt", "best_genome_predictive_evolution.pkl")

    json_path = "predictive_evolution_history.json"
    with open(json_path, "w") as f:
        json.dump(predictive_evolutions_history, f, indent=4)
