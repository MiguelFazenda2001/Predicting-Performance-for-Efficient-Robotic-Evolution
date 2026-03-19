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

# Predictive Model
from transformer import EpisodeTransformer
import torch

class GenerationTracker(neat.reporting.BaseReporter):
    def start_generation(self, generation):
        global current_generation
        current_generation = generation

class PredictiveEvolution:
    def __init__(self, num_generations, n_episodes_to_predict, total_n_episodes, model_path, input_dim, x_mean, x_std, y_mean, y_std, step_limit=500):
        self.num_generations = num_generations
        self.n_episodes_to_predict = n_episodes_to_predict
        self.total_n_episodes = total_n_episodes
        self.step_limit = step_limit
        self.fitness_history = []

        # Load predictive model
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.predictive_model = EpisodeTransformer(input_dim=input_dim).to(self.device)
        self.predictive_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.predictive_model.eval()

        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def evaluate_genome(self, genome, config):

        environment = gym.make("LunarLander-v3", continuous=True, enable_wind=True, wind_power=15.0, turbulence_power=1.5)

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        success_count = 0
        total_rewards = []
        all_obervations = []
        all_actions = []

        for episode_id in range(self.total_n_episodes):
            obs, _ = environment.reset()
            total_reward = 0
            obervations, actions = [], []

            for step in range(self.step_limit):
                obervations.append(obs)
                action = np.clip(net.activate(obs), -1.0, 1.0)
                actions.append(action)
                obs, reward, terminated, truncated, _ = environment.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            total_rewards.append(total_reward)
            all_obervations.append(obervations)
            all_actions.append(actions)

            if total_reward >= 200:
                success_count += 1

        environment.close()

        # Choose wich episodes to predict on random n

        seqs = []
        masks = []
        for i in range(self.n_episodes_to_predict):
            idx = np.random.randint(0, self.total_n_episodes)
            seq = np.concatenate([all_obervations[idx], all_actions[idx]], axis=1)

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

        # Normalization

        X = np.array(seqs)
        M = np.array(masks)

        mask_exp = M[..., None]

        X = (X - self.x_mean) / self.x_std
        X *= mask_exp

        # Fazer a previsão

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
            M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0).to(self.device)

            predictions = self.predictive_model(X_tensor, M_tensor)
            predictions = predictions.cpu().numpy()

        return predictions, np.mean(total_rewards), success_count / self.total_n_episodes


    def eval_population(self, population, config):
        for genome_id, genome in population:
            time_start = time.time()
            predictions, fitness, success_rate = self.evaluate_genome(genome, config)
            time_end = time.time()
            genome.fitness = predictions[0][0] * 100 # To add depth
            self.fitness_history.append({
                "generation": current_generation,
                "genome_id": genome_id,
                "prediction(used_fitness = pred * 100)": float(predictions[0][0]),
                "fitness": fitness,
                "success_rate": success_rate,
                "evaluation_time": (time_end - time_start)/self.total_n_episodes * self.n_episodes_to_predict
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