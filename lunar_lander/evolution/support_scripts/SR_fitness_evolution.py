import gymnasium as gym
import numpy as np
import neat
import os
import pickle
import pandas as pd
import json
import glob
import argparse
import h5py
from datetime import datetime
import time

class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation

class SRFitnessEvolution:
    def __init__(self, num_generations, n_episodes_for_fitness, total_n_episodes, std = None, step_limit=500):
        self.num_generations = num_generations
        self.n_episodes_for_fitness = n_episodes_for_fitness
        self.total_n_episodes = total_n_episodes
        self.step_limit = step_limit
        self.std = std
        self.fitness_history = []

    def evaluate_genome(self, genome, config):

        environment = gym.make("LunarLander-v3", continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5)

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        success_count = 0
        total_rewards = []
        success_results = []

        for episode_id in range(self.total_n_episodes):
            obs, _ = environment.reset()
            total_reward = 0

            for step in range(self.step_limit):
                action = np.clip(net.activate(obs), -1.0, 1.0)
                obs, reward, terminated, truncated, _ = environment.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            total_rewards.append(total_reward)

            if total_reward >= 200:
                success_count += 1
                success_results.append(1)
            else:
                success_results.append(0)

            
        environment.close()

        episodes_for_fitness = np.random.choice(self.total_n_episodes, self.n_episodes_for_fitness, replace=False)
        used_fitness = np.mean([total_rewards[i] for i in episodes_for_fitness])

        used_success_rate = np.mean([success_results[i] for i in episodes_for_fitness])

        return used_fitness, used_success_rate, success_count / self.total_n_episodes, np.mean(total_rewards)


    def eval_population(self, population, config):
        for genome_id, genome in population:
            time_start = time.time()
            fitness_for_evolution, success_rate_for_evolution, success_rate, fitness_all_episodes = self.evaluate_genome(genome, config)
            time_end = time.time()
            if self.std is not None:
                success_rate = round(success_rate, 1)

                std_value = self.std.get(success_rate, 0.0)

                noise = np.random.normal(loc=0.0, scale=std_value)

                genome.fitness = (
                    (success_rate_for_evolution + noise) * 100 #+ fitness_for_evolution * 0.01 + noise
                )

            else:    
                genome.fitness = success_rate_for_evolution * 100 + fitness_for_evolution * 0.01 # Just to have a small diference for the beginning
            self.fitness_history.append({
                "generation": current_generation,
                "genome_id": genome_id,
                "fitness_for_evolution": fitness_for_evolution,
                "fitness": fitness_all_episodes,
                "success_rate_for_evolution": success_rate_for_evolution,
                "success_rate": success_rate,
                "evaluation_time": (time_end - time_start)/self.total_n_episodes * self.n_episodes_for_fitness
            })


    def evolve(self, config_file, model_path):
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )
        population = neat.Population(config)

        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())
        population.add_reporter(GenerationTracker())

        best_genome = population.run(lambda genomes, config: self.eval_population(genomes, config), self.num_generations)

        with open(model_path, "wb") as f:
            pickle.dump(best_genome, f)

        return self.fitness_history