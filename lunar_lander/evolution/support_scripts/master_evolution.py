from fitness_evolution import FitnessEvolution
from SR_fitness_evolution import SRFitnessEvolution
from predictive_evolution_1_ep_tcn import PredictiveEvolution
import json
import time
import numpy as np
import os

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

    """
    os.makedirs(f"../neat_models/Predictive_VS_Fitness/30_total_episodes_10_episodes_sr_fitness_evolution", exist_ok=True)
    os.makedirs(f"../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_10_episodes_sr_fitness_evolution", exist_ok=True)
    
    for i in range(25):
        print(f"Running sr fitness evolution {i+1}/25")

        fitness_evolver = SRFitnessEvolution(num_generations=150, n_episodes_for_fitness=3, total_n_episodes=30, std=None, step_limit=STEP_LIMIT)

        fitness_evolutions_history = fitness_evolver.evolve("config-feedforward.txt", f"../neat_models/Predictive_VS_Fitness/30_total_episodes_10_episodes_sr_fitness_evolution/best_genome_sr_fitness_evolution_{i+1}.pkl")

        json_path = f"../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_10_episodes_sr_fitness_evolution/sr_fitness_evolution_history_{i+1}.json"
        with open(json_path, "w") as f:
            json.dump(fitness_evolutions_history, f, indent=4)

    """
    x_mean = np.load("../files_for_predictive_model/1_ep_tcn/x_mean.npy")
    x_std = np.load("../files_for_predictive_model/1_ep_tcn/x_std.npy")
    y_mean = np.load("../files_for_predictive_model/1_ep_tcn/y_mean.npy")
    y_std = np.load("../files_for_predictive_model/1_ep_tcn/y_std.npy")

    duration_multiplier = [0.0]

    for w in duration_multiplier:

        # folder_name_w =  "sr_20"

        os.makedirs(f"../neat_models/Predictive_VS_Fitness/30_total_episodes_1_episodes_predictive_tcn_evolution", exist_ok=True)
        os.makedirs(f"../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_1_episodes_predictive_tcn_evolution", exist_ok=True)

        for i in range(25):
            print(f"Running predictive evolution {i+1}/25")

            predictive_evolver = PredictiveEvolution(num_generations=150, n_episodes_to_predict=1, total_n_episodes=30, model_path="../files_for_predictive_model/1_ep_tcn/residual_tcn_model.pth", input_dim=10, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, step_limit=STEP_LIMIT)

            predictive_evolutions_history = predictive_evolver.evolve("config-feedforward.txt", f"../neat_models/Predictive_VS_Fitness/30_total_episodes_1_episodes_predictive_tcn_evolution/best_genome_predictive_tcn_evolution_{i+1}.pkl")

            json_path = f"../evolutionary_history/Predictive_VS_Fitness/30_total_episodes_1_episodes_predictive_tcn_evolution/predictive_tcn_evolution_history_{i+1}.json"
            with open(json_path, "w") as f:
                json.dump(predictive_evolutions_history, f, indent=4)

    
    
    


