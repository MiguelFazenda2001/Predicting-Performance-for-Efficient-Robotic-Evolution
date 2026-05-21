from fitness_evolution import FitnessEvolution
from predictive_evolution import PredictiveEvolution
import json
import time
import numpy as np
import os

STEP_LIMIT = 150


if __name__ == "__main__":

    """    
    os.makedirs(f"../neat_models/30_total_episodes_3_episodes_fitness_evolution", exist_ok=True)
    os.makedirs(f"../evolutionary_history/30_total_episodes_3_episodes_fitness_evolution", exist_ok=True)
    
    for i in range(25):
        print(f"Running fitness evolution {i+1}/25")

        fitness_evolver = FitnessEvolution(num_generations=250, n_episodes_for_fitness=3, total_n_episodes=30, fitness_history_path=f"../evolutionary_history/30_total_episodes_3_episodes_fitness_evolution/fitness_evolution_history_{i+1}.json", std=None, step_limit=STEP_LIMIT)

        fitness_evolver.evolve("config-feedforward.txt", f"../neat_models/30_total_episodes_3_episodes_fitness_evolution/best_genome_fitness_evolution_{i+1}.pkl")

    """
    x_mean = np.load("../files_for_predictive_model/3_ep/x_mean.npy")
    x_std = np.load("../files_for_predictive_model/3_ep/x_std.npy")
    y_mean = np.load("../files_for_predictive_model/3_ep/y_mean.npy")
    y_std = np.load("../files_for_predictive_model/3_ep/y_std.npy")

    duration_multiplier = [0.0]

    for w in duration_multiplier:

        # folder_name_w =  "sr_20"

        os.makedirs(f"../neat_models/30_total_episodes_3_episodes_predictive_evolution_300_gens", exist_ok=True)
        os.makedirs(f"../evolutionary_history/30_total_episodes_3_episodes_predictive_evolution_300_gens", exist_ok=True)
        os.makedirs(f"../evolution_checkpoints/30_total_episodes_3_episodes_predictive_evolution_300_gens", exist_ok=True)

        for i in range(25):
            print(f"Running predictive evolution {i+1}/25")

            predictive_evolver = PredictiveEvolution(num_generations=300, n_episodes_to_predict=3, total_n_episodes=30, model_path="../files_for_predictive_model/3_ep/predictive_model_3_ep.pth", input_dim=17, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, fitness_history_path=f"../evolutionary_history/30_total_episodes_3_episodes_predictive_evolution_300_gens/predictive_evolution_history_{i+1}.json", checkpoint_path=f"../evolution_checkpoints/30_total_episodes_3_episodes_predictive_evolution_300_gens/checkpoint_{i+1}_", step_limit=STEP_LIMIT)

            predictive_evolver.evolve("config-feedforward.txt", f"../neat_models/30_total_episodes_3_episodes_predictive_evolution_300_gens/best_genome_predictive_evolution_{i+1}.pkl")

    
    
    


