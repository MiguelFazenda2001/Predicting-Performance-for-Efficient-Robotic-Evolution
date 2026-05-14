import gymnasium as gym
import gymnasium_robotics
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
from multiprocessing import Pool, cpu_count

class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation

class FitnessEvolution:
    def __init__(self, num_generations, n_episodes_for_fitness, total_n_episodes, std=None, duration_multiplier = 0, step_limit=500):
        self.num_generations = num_generations
        self.n_episodes_for_fitness = n_episodes_for_fitness
        self.total_n_episodes = total_n_episodes
        self.std = std
        self.duration_multiplier = duration_multiplier
        self.step_limit = step_limit
        self.fitness_history = []
        self.range_vals, self.min_vals = self.load_limits()

    def load_limits(self, path="observation_limits_13_obs.json"):
        with open(path, "r") as f:
            data = json.load(f)

        min_vals = np.array(data["min"])
        max_vals = np.array(data["max"])

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-8

        return range_vals, min_vals


    def evaluate_genome(self, genome, config):

        env = gym.make('FetchReachDense-v4', self.step_limit)

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        success_count = 0
        total_rewards = []
        episode_durations = []
        final_rewards = []

        for episode_id in range(self.total_n_episodes):
            dict_obs, _ = env.reset()
            total_reward = 0
            episode_duration = 0

            for step in range(self.step_limit):
                obs = np.concatenate([dict_obs["observation"],dict_obs["desired_goal"] - dict_obs["achieved_goal"]])
                obs = self.normalize_observation(obs)

                action = np.clip(net.activate(obs), -1.0, 1.0)

                dict_obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward 

                if terminated or truncated:
                    break

            if reward >= -0.05:
                success_count += 1
                total_reward += 100

            total_rewards.append(total_reward)
            final_rewards.append(reward)
            episode_durations.append(episode_duration)

            
        env.close()

        episodes_for_fitness = np.random.choice(self.total_n_episodes, self.n_episodes_for_fitness, replace=False)
        used_fitness = np.mean([final_rewards[i] for i in episodes_for_fitness]) * 10
        avg_ep_duration = np.mean([episode_durations[i] for i in episodes_for_fitness])
        avg_ep_duration_all_episodes = np.mean(episode_durations)

        return used_fitness, success_count / self.total_n_episodes, np.mean(final_rewards), avg_ep_duration, avg_ep_duration_all_episodes


    def eval_population(self, population, config):
        
        
        genome_map = dict(population)

        args = [(genome, config) for _, genome in population]

        with Pool(processes=50) as pool:
            results = pool.map(self.eval_worker, args)
        
        for genome_id,  reward, success_rate, fitness_all_episodes, average_episode_duration, avg_ep_duration_all_episodes, evaluation_time in results:
            
            genome_map[genome_id].fitness = reward - (self.duration_multiplier * (average_episode_duration * 0.2)) # *0.2 to scale avg duration to 0-100
            
            self.fitness_history.append({
                "generation": current_generation,
                "genome_id": genome_id,
                "used_fitness": reward,
                "fitness": fitness_all_episodes,
                "success_rate": success_rate,
                "average_episode_duration": average_episode_duration,
                "avg_ep_duration_all_episodes": avg_ep_duration_all_episodes,
                "evaluation_time": evaluation_time
            })

    def eval_worker(self, args):
        genome, config = args
        time_start = time.time()
        reward, success_rate, fitness_all_episodes, average_episode_duration, avg_ep_duration_all_episodes = self.evaluate_genome(genome, config)
        time_end = time.time()
        evaluation_time = time_end - time_start
        return genome.key, reward, success_rate, fitness_all_episodes, average_episode_duration, avg_ep_duration_all_episodes, evaluation_time


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

    def normalize_observation(self, obs):
        # scale to [0,1]
        norm = (obs - self.min_vals) / self.range_vals

        # scale to [-1,1]
        norm = norm * 2.0 - 1.0

        return norm
