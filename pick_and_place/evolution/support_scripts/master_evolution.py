from fitness_evolution import FitnessEvolution
from predictive_evolution import PredictiveEvolution
import json
import time
import numpy as np
import os

STEP_LIMIT = 150


if __name__ == "__main__":

    """
    os.makedirs(f"../neat_models/30_total_episodes_10_episodes_fitness_evolution_300_gens", exist_ok=True)
    os.makedirs(f"../evolutionary_history/30_total_episodes_10_episodes_fitness_evolution_300_gens", exist_ok=True)
    os.makedirs(f"../evolution_checkpoints/30_total_episodes_10_episodes_fitness_evolution_300_gens", exist_ok=True)

    
    for i in range(20):
        if i <= 16:
            continue
        elif i == 17:
            gens = 69
        else:
            continue
        print(f"Running fitness evolution {i+1}/25")
        fitness_evolver = FitnessEvolution(num_generations=gens, n_episodes_for_fitness=1, total_n_episodes=30, fitness_history_path=f"../evolutionary_history/30_total_episodes_10_episodes_fitness_evolution_300_gens/fitness_evolution_history_{i+1}.json", checkpoint_path=f"../evolution_checkpoints/30_total_episodes_10_episodes_fitness_evolution_300_gens/checkpoint_{i+1}_", std=None, step_limit=STEP_LIMIT)

        fitness_evolver.evolve("config-feedforward.txt", f"../neat_models/30_total_episodes_10_episodes_fitness_evolution_300_gens/best_genome_fitness_evolution_{i+1}.pkl")

    
    """
    x_mean = np.load("../files_for_predictive_model/1_ep/x_mean.npy")
    x_std = np.load("../files_for_predictive_model/1_ep/x_std.npy")
    y_mean = np.load("../files_for_predictive_model/1_ep/y_mean.npy")
    y_std = np.load("../files_for_predictive_model/1_ep/y_std.npy")

    duration_multiplier = [0.0]

    for w in duration_multiplier:

        os.makedirs(f"../neat_models/30_total_episodes_1_episodes_predictive_evolution_300_gens", exist_ok=True)
        os.makedirs(f"../evolutionary_history/30_total_episodes_1_episodes_predictive_evolution_300_gens", exist_ok=True)
        os.makedirs(f"../evolution_checkpoints/30_total_episodes_1_episodes_predictive_evolution_300_gens", exist_ok=True)

        for i in range(20):
            if i <= 5:
                continue
            elif i == 6:
                gens = 143
            else:
                gens = 300

            print(f"Running predictive evolution {i+1}/20")

            predictive_evolver = PredictiveEvolution(num_generations=gens, n_episodes_to_predict=1, total_n_episodes=30, model_path="../files_for_predictive_model/1_ep/predictive_model_1_ep.pth", input_dim=32, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, fitness_history_path=f"../evolutionary_history/30_total_episodes_1_episodes_predictive_evolution_300_gens/predictive_evolution_history_{i+1}.json", checkpoint_path=f"../evolution_checkpoints/30_total_episodes_1_episodes_predictive_evolution_300_gens/checkpoint_{i+1}_", step_limit=STEP_LIMIT)

            predictive_evolver.evolve("config-feedforward.txt", f"../neat_models/30_total_episodes_1_episodes_predictive_evolution_300_gens/best_genome_predictive_evolution_{i+1}.pkl")

    
    
    


