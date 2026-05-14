from fitness_evolution import FitnessEvolution
from predictive_evolution import PredictiveEvolution
import json
import time
import numpy as np
import os

STEP_LIMIT = 150


if __name__ == "__main__":

    """
    os.makedirs(f"../neat_models/10_episodes_fitness_evolution", exist_ok=True)
    os.makedirs(f"../evolutionary_history/10_episodes_fitness_evolution", exist_ok=True)
    
    for i in range(24):
        print(f"Running fitness evolution {i+2}/25")

        fitness_evolver = FitnessEvolution(num_generations=300, n_episodes_for_fitness=10, total_n_episodes=10, std=None, step_limit=STEP_LIMIT)

        fitness_evolutions_history = fitness_evolver.evolve("config-feedforward.txt", f"../neat_models/10_episodes_fitness_evolution/best_genome_10_episodes_fitness_evolution_{i+2}.pkl")

        json_path = f"../evolutionary_history/10_episodes_fitness_evolution/10_episodes_fitness_evolution_history_{i+2}.json"
        with open(json_path, "w") as f:
            json.dump(fitness_evolutions_history, f, indent=4)

    """
    x_mean = np.load("../files_for_predictive_model/1_ep/x_mean.npy")
    x_std = np.load("../files_for_predictive_model/1_ep/x_std.npy")
    y_mean = np.load("../files_for_predictive_model/1_ep/y_mean.npy")
    y_std = np.load("../files_for_predictive_model/1_ep/y_std.npy")

    duration_multiplier = [0.0]

    for w in duration_multiplier:

        # folder_name_w =  "sr_20"

        os.makedirs(f"../neat_models//1_episode_predictive_evolution", exist_ok=True)
        os.makedirs(f"../evolutionary_history/1_episode_predictive_evolution", exist_ok=True)

        for i in range(25):
            print(f"Running predictive evolution {i+2}/25")

            predictive_evolver = PredictiveEvolution(num_generations=1, n_episodes_to_predict=1, total_n_episodes=10, model_path="../files_for_predictive_model/1_ep/predictive_model_1_ep.pth", input_dim=17, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, step_limit=STEP_LIMIT)

            predictive_evolutions_history = predictive_evolver.evolve("config-feedforward.txt", f"../neat_models/1_episode_predictive_evolution/best_genome_1_episode_predictive_evolution_{i+2}.pkl")

            json_path = f"../evolutionary_history/1_episode_predictive_evolution/1_episode_predictive_evolution_history_{i+2}.json"
            with open(json_path, "w") as f:
                json.dump(predictive_evolutions_history, f, indent=4)
    
    
    
    


