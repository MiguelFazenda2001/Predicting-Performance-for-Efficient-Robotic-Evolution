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
import multiprocessing as mp

from transformer_1_ep import EpisodeTransformer
import torch

class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation

class PredictiveEvolution:
    def __init__(self, num_generations, n_episodes_to_predict, total_n_episodes, model_path, input_dim, x_mean, x_std, y_mean, y_std, duration_multiplier = 0, step_limit=150):
        mp.set_start_method("spawn", force=True)
        self.num_generations = num_generations
        self.n_episodes_to_predict = n_episodes_to_predict
        self.total_n_episodes = total_n_episodes
        self.duration_multiplier = duration_multiplier
        self.step_limit = step_limit
        self.fitness_history = []
        self.range_vals, self.min_vals = self.load_limits()

        # Load predictive model
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.predictive_model = EpisodeTransformer(input_dim=input_dim).to(self.device)
        self.predictive_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.predictive_model.eval()

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

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
        all_observations = []
        all_actions = []

        for episode_id in range(self.total_n_episodes):
            dict_obs, _ = env.reset()
            total_reward = 0
            episode_duration = 0
            observations, actions = [], []

            for step in range(self.step_limit):
                obs = np.concatenate([dict_obs["observation"],dict_obs["desired_goal"] - dict_obs["achieved_goal"]])
                obs = self.normalize_observation(obs)
                observations.append(obs)

                action = np.clip(net.activate(obs), -1.0, 1.0)
                actions.append(action)

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
            all_observations.append(observations)
            all_actions.append(actions)

        env.close()

        seqs = []
        masks = []

        idxs = np.random.choice(self.total_n_episodes, self.n_episodes_to_predict, replace=False)
        
        for idx in idxs:
            seq = np.concatenate([all_observations[idx], all_actions[idx]], axis=1)

            mask = np.ones(self.step_limit, dtype=np.float32)

            length = len(seq)
            if length >= self.step_limit:
                seq = seq[:self.step_limit]
                mask = mask[:self.step_limit]
            else:
                pad = np.zeros((self.step_limit - length, seq.shape[1]), dtype=np.float32)
                seq = np.vstack([seq, pad])
                mask[length:] = 0.0

            seqs.append(seq)
            masks.append(mask)

        X = np.array(seqs)
        M = np.array(masks)

        mask_exp = M[..., None]

        X = (X - self.x_mean) / self.x_std
        X *= mask_exp

        # Fazer a previsão
        used_fitness = None
        avg_ep_duration = np.mean([episode_durations[i] for i in idxs])
        avg_ep_duration_all_episodes = np.mean(episode_durations)

        return used_fitness, success_count / self.total_n_episodes, np.mean(final_rewards), avg_ep_duration, avg_ep_duration_all_episodes, X, M


    def eval_population(self, population, config):
        genome_map = dict(population)

        args = [(genome, config) for _, genome in population]

        with mp.Pool(processes=50) as pool:
            results = pool.map(self.eval_worker, args)
        
        for genome_id,  reward, success_rate, fitness_all_episodes, average_episode_duration, avg_ep_duration_all_episodes, evaluation_time, X, M in results:
            
            time_start = time.time()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
                M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0).to(self.device)

                predictions = self.predictive_model(X_tensor, M_tensor)
                predictions = predictions.cpu().numpy()
                used_fitness = predictions[0][0] * 100  

            genome_map[genome_id].fitness = used_fitness - (self.duration_multiplier * (average_episode_duration * 0.2)) # *0.2 to scale avg duration to 0-100
            time_end = time.time()
            evaluation_time += (time_end - time_start)

            self.fitness_history.append({
                "generation": current_generation,
                "genome_id": genome_id,
                "used_fitness": float(used_fitness),
                "fitness": fitness_all_episodes,
                "success_rate": success_rate,
                "average_episode_duration": average_episode_duration,
                "avg_ep_duration_all_episodes": avg_ep_duration_all_episodes,
                "evaluation_time": evaluation_time
            })

    def eval_worker(self, args):
        genome, config = args
        time_start = time.time()
        reward, success_rate, fitness_all_episodes, average_episode_duration, avg_ep_duration_all_episodes, X, M = self.evaluate_genome(genome, config)
        time_end = time.time()
        evaluation_time = time_end - time_start
        return genome.key, reward, success_rate, fitness_all_episodes, average_episode_duration, avg_ep_duration_all_episodes, evaluation_time, X, M


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
