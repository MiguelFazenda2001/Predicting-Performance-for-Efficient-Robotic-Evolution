import numpy as np
import gymnasium as gym
import gymnasium_robotics
import cma
import json
import numpy as np
from multiprocessing import Pool
import h5py
import gc
import time
import os
import pickle
import torch
from transformer_1_ep import EpisodeTransformer

"""

Normalize distances to [0,1] per episode

"""


# ── Configuration ──────────────────────────────────────────────
ENV_ID        = "FetchPushDense-v4"
TOTAL_N_EPISODES    = 30       # evaluations per candidate
N_EPISODES_FOR_FITNESS = 1
POP_SIZE      = 500       # CMA-ES lambda (population size)
MAX_GENS      = 500      # generations budget
SIGMA0        = 0.5      # initial step size
ALPHA         = 0.3      # distance penalty weight in fitness
HIDDEN_64     = 64       # hidden units in the controller MLP
HIDDEN_32     = 32       # hidden units in the controller MLP
HIDDEN_16     = 16       # hidden units in the controller MLP

_worker_env = None
_worker_controller = None
_range_vals = None
_min_vals = None

def init_worker(obs_dim, act_dim, range_vals, min_vals, x_mean, x_std):
    global _worker_env, _worker_controller
    global _range_vals, _min_vals, _x_mean, _x_std

    gym.register_envs(gymnasium_robotics)

    _worker_env = gym.make(ENV_ID, max_episode_steps=150)
    _worker_controller = MLPController(obs_dim, act_dim)

    _range_vals = range_vals
    _min_vals = min_vals

    _x_mean = x_mean
    _x_std = x_std


def evaluate_worker(params):
    time_start = time.time()
    fitness, success_rate, avg_ep_duration, avg_ep_duration_all_episodes, X, M = evaluate(params, _worker_env, _worker_controller, _range_vals, _min_vals, _x_mean, _x_std)
    time_end = time.time()
    evaluation_time = time_end - time_start
    evaluation_time = (evaluation_time  / TOTAL_N_EPISODES) * N_EPISODES_FOR_FITNESS
    return fitness, success_rate, avg_ep_duration, avg_ep_duration_all_episodes, evaluation_time, X, M

# ── Controller: small MLP obs → action ─────────────────────────
class MLPController:
    def __init__(self, obs_dim, act_dim, hidden_64=HIDDEN_64, hidden_32=HIDDEN_32, hidden_16=HIDDEN_16):
        self.shapes = [
            (hidden_64, obs_dim), (hidden_64,),    # W1, b1
            (hidden_64, hidden_64),      (hidden_64,),    # W2, b2
            (hidden_32, hidden_64),      (hidden_32,),    # W3, b3
            (act_dim, hidden_32), (act_dim,) # W5, b5
        ]
        self.n_params = sum(np.prod(s) for s in self.shapes)

    def set_weights(self, params):
        self.params = params

    def act(self, obs):
        x = obs.astype(np.float32)

        idx = 0
        layers = []
        for shape in self.shapes:
            size = np.prod(shape)
            layers.append(self.params[idx:idx+size].reshape(shape))
            idx += size

        W1, b1, W2, b2, W3, b3, W4, b4 = layers

        x = np.tanh(W1 @ x + b1)
        x = np.tanh(W2 @ x + b2)
        x = np.tanh(W3 @ x + b3) 
        x = np.tanh(W4 @ x + b4)        # Output

        return x

# ── Fitness function ────────────────────────────────────────────
def evaluate(params, env, controller, range_vals, min_vals, x_mean, x_std, total_n_episodes=TOTAL_N_EPISODES, n_episodes_to_predict=N_EPISODES_FOR_FITNESS):
    """
    Returns scalar fitness (higher = better).
    fitness = success_rate - ALPHA * mean_goal_distance
    """
    controller.set_weights(params)
    success_count = 0
    total_rewards = []
    episode_durations = []
    final_rewards = []
    successes = []
    all_observations = []
    all_actions = []

    idxs = np.random.choice(total_n_episodes, n_episodes_to_predict, replace=False)

    for episode_id in range(total_n_episodes):
        obs_dict, _ = env.reset()

        total_reward = 0
        success_step = None
        observations, actions = [], []

        for step in range(env._max_episode_steps):
            obs = np.concatenate([obs_dict["observation"], obs_dict["desired_goal"] - obs_dict["achieved_goal"]])
            obs = normalize_observation(obs, range_vals, min_vals)

            action = controller.act(obs)

            if episode_id in idxs:
                observations.append(obs)
                actions.append(action)

            obs_dict, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if reward >= -0.05 and success_step is None:
                success_step = step
            elif reward < -0.05:
                success_step = None

            if terminated or truncated:
                break

        if reward > -0.05:  
            success_count += 1
            success = 1
        else:
            success = 0

        if success_step is None:
            success_step = env._max_episode_steps - 1

        total_rewards.append(total_reward)
        final_rewards.append(reward)

        duration = success_step / env.metadata["render_fps"]
        episode_durations.append(duration)

        successes.append(success)
        if episode_id in idxs:
            all_observations.append(observations)
            all_actions.append(actions)

    seqs = []
    masks = []
    
    for i in range(n_episodes_to_predict):
        seq = np.concatenate([all_observations[i], all_actions[i]], axis=1)

        mask = np.ones(env._max_episode_steps, dtype=np.float32)

        length = len(seq)
        if length >= env._max_episode_steps:
            seq = seq[:env._max_episode_steps]
            mask = mask[:env._max_episode_steps]
        else:
            pad = np.zeros((env._max_episode_steps - length, seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
            mask[length:] = 0.0

        seqs.append(seq)
        masks.append(mask)

    X = np.array(seqs)
    M = np.array(masks)

    mask_exp = M[..., None]

    X = (X - x_mean) / x_std
    X *= mask_exp

    # Fazer a previsão
    used_fitness = None
    avg_ep_duration = np.mean([episode_durations[i] for i in idxs])
    avg_ep_duration_all_episodes = np.mean(episode_durations)

    fitness = np.mean([final_rewards[i] for i in idxs]) * 100 + (sum([successes[i] for i in idxs]) / n_episodes_to_predict) * 100
    avg_ep_duration = np.mean([episode_durations[i] for i in idxs])
    avg_ep_duration_all_episodes = np.mean(episode_durations)

    env.close()

    return fitness, success_count / total_n_episodes, avg_ep_duration, avg_ep_duration_all_episodes, X, M  # CMA-ES minimises, so we negate below

def load_limits(path="observation_limits_push.json"):
    with open(path, "r") as f:
        data = json.load(f)

    min_vals = np.array(data["min"])
    max_vals = np.array(data["max"])

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8

    return range_vals, min_vals

def normalize_observation(obs, range_vals, min_vals):
    # scale to [0,1]
    norm = (obs - min_vals) / range_vals

    # scale to [-1,1]
    norm = norm * 2.0 - 1.0

    return norm

def save_checkpoint(es, generation, best_fitness, best_params, evo):
    checkpoint = {
        "es": es,
        "generation": generation,
        "best_fitness": best_fitness,
        "best_params": best_params,
    }

    tmp = f"../evolution_checkpoints/predictive_1_ep_checkpoint_{evo}.tmp"
    final = f"../evolution_checkpoints/predictive_1_ep_checkpoint_{evo}.pkl"

    with open(tmp, "wb") as f:
        pickle.dump(checkpoint, f)

    os.replace(tmp, final)

# ── Main evolution loop ─────────────────────────────────────────
def main(evo=None, model_path=None):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, max_episode_steps=150)

    # Infer obs and action dims from the environment
    obs_dict, _ = env.reset()
    obs_dim = len(obs_dict["observation"]) + len(obs_dict["desired_goal"] - obs_dict["achieved_goal"])
    act_dim = env.action_space.shape[0]

    # Load observation limits
    range_vals, min_vals = load_limits()
    x_mean = np.load("../files_for_predictive_model/1_ep/x_mean.npy")
    x_std = np.load("../files_for_predictive_model/1_ep/x_std.npy")

    n_workers = 50

    pool = Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(
            obs_dim,
            act_dim,
            range_vals,
            min_vals,
            x_mean,
            x_std
        )
    )

    controller = MLPController(obs_dim, act_dim)
    print(f"Controller params: {controller.n_params}")
    print(f"Obs dim: {obs_dim} | Act dim: {act_dim}")

    if os.path.exists(f"../evolution_checkpoints/predictive_1_ep_checkpoint_{evo}.pkl"):
        print("Loading checkpoint...")

        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        es = checkpoint["es"]
        generation = checkpoint["generation"]
        best_fitness = checkpoint["best_fitness"]
        best_params = checkpoint["best_params"]

    else:
        opts = cma.CMAOptions()
        opts["popsize"] = POP_SIZE
        opts["maxiter"] = MAX_GENS
        opts["verbose"] = 1
        opts["tolx"] = 1e-5
        opts["tolfun"] = 1e-5
        opts["CMA_elitist"] = True

        x0 = np.zeros(controller.n_params)
        es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)
        generation = 0
        best_fitness = -np.inf
        best_params = None

    while not es.stop():
        time_start_gen = time.time()
        solutions = es.ask()          # sample λ candidates
        fitnesses = []
        data = []
        i = 0

        results = pool.map(evaluate_worker, solutions, chunksize=10)

        prediction_time_start = time.time()

        # Load predictive model
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        predictive_model = EpisodeTransformer(input_dim=32).to(device)
        predictive_model.load_state_dict(torch.load(model_path, map_location=device))
        predictive_model.eval()

        all_X = np.stack([r[5] for r in results])
        all_M = np.stack([r[6] for r in results])

        with torch.no_grad():
            X_tensor = torch.tensor(all_X, dtype=torch.float32).to(device)
            M_tensor = torch.tensor(all_M, dtype=torch.float32).to(device)

            predictions = predictive_model(X_tensor, M_tensor)
            predictions = predictions.cpu().numpy()

        prediction_time_end = time.time()
        prediction_time = (prediction_time_end - prediction_time_start) / POP_SIZE

        fitnesses = []
        data = []

        for i, (f, success_rate, avg_ep_duration, avg_ep_duration_all_episodes, evaluation_time, X, M) in enumerate(results):
            fit = predictions[i][0] * 100
            fitnesses.append(fit)
            evaluation_time += prediction_time

            data.append({
                "generation": generation,
                "solution_id": i,
                "fitness": -f,
                "prediction_fitness": float(fit),
                "success_rate": success_rate,
                "avg_ep_duration": avg_ep_duration,
                "avg_ep_duration_all_episodes": avg_ep_duration_all_episodes,
                "evaluation_time": evaluation_time
            })

            if fit > best_fitness:
                best_fitness = fit
                best_params = solutions[i].copy()

        # CMA-ES minimises → negate fitness
        es.tell(solutions, [-fit for fit in fitnesses])
        es.logger.add()

        generation += 1
        mean_f = -np.mean(fitnesses)

        with open(f"../evolutionary_history/30_total_episodes_1_episodes_predictive_cma/cma_predictive_evo_{evo}.jsonl", "a") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        if generation % 10 == 0:
            save_checkpoint(
                es,
                generation,
                best_fitness,
                best_params,
                evo
            )
        time_end_gen = time.time()
        gen_time = time_end_gen - time_start_gen
        print(f"Gen {generation:4d} | best={best_fitness:.3f} | mean={mean_f:.3f} | sigma={es.sigma:.4f} | time={gen_time} seconds")

    pool.close()
    pool.join()
    pool.terminate()

    env.close()

    # Save best controller
    np.save(f"../cma_models/best_controller_predictive_1_ep_{evo}.npy", best_params)
    print(f"\nDone. Best fitness: {best_fitness:.3f}")
    print(f"Saved weights to models/best_controller_predictive_1_ep_{evo}.npy")


    del es
    del pool
    del env

    return best_params


if __name__ == "__main__":
    for i in range(5):
        main(evo=i, model_path = "../files_for_predictive_model/1_ep/predictive_model_1_ep.pth")
        gc.collect()