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

"""

Normalize distances to [0,1] per episode

"""


# ── Configuration ──────────────────────────────────────────────
ENV_ID        = "FetchPushDense-v4"
TOTAL_N_EPISODES    = 30       # evaluations per candidate
N_EPISODES_FOR_FITNESS = 10
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

def init_worker(obs_dim, act_dim, range_vals, min_vals):
    global _worker_env, _worker_controller
    global _range_vals, _min_vals

    gym.register_envs(gymnasium_robotics)

    _worker_env = gym.make(ENV_ID, max_episode_steps=150)
    _worker_controller = MLPController(obs_dim, act_dim)

    _range_vals = range_vals
    _min_vals = min_vals


def evaluate_worker(params):
    time_start = time.time()
    fitness, success_rate, avg_ep_duration, avg_ep_duration_all_episodes = evaluate(params, _worker_env, _worker_controller, _range_vals, _min_vals,)
    time_end = time.time()
    evaluation_time = time_end - time_start
    evaluation_time = (evaluation_time  / TOTAL_N_EPISODES) * N_EPISODES_FOR_FITNESS
    return fitness, success_rate, avg_ep_duration, avg_ep_duration_all_episodes, evaluation_time

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
def evaluate(params, env, controller, range_vals, min_vals, total_n_episodes=TOTAL_N_EPISODES, n_episodes_for_fitness=N_EPISODES_FOR_FITNESS):
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

    for _ in range(total_n_episodes):
        obs_dict, _ = env.reset()

        total_reward = 0
        success_step = None

        for step in range(env._max_episode_steps):
            obs = np.concatenate([obs_dict["observation"], obs_dict["desired_goal"] - obs_dict["achieved_goal"]])
            obs = normalize_observation(obs, range_vals, min_vals)

            action = controller.act(obs)

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
    
    env.close()

    episodes_for_fitness = np.random.choice(total_n_episodes, n_episodes_for_fitness, replace=False)
    fitness = np.mean([final_rewards[i] for i in episodes_for_fitness]) * 100 + (sum([successes[i] for i in episodes_for_fitness]) / n_episodes_for_fitness) * 100
    avg_ep_duration = np.mean([episode_durations[i] for i in episodes_for_fitness])
    avg_ep_duration_all_episodes = np.mean(episode_durations)

    return fitness, success_count / total_n_episodes, avg_ep_duration, avg_ep_duration_all_episodes  # CMA-ES minimises, so we negate below

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

    tmp = f"../evolution_checkpoints/checkpoint_{evo}.tmp"
    final = f"../evolution_checkpoints/checkpoint_{evo}.pkl"

    with open(tmp, "wb") as f:
        pickle.dump(checkpoint, f)

    os.replace(tmp, final)

# ── Main evolution loop ─────────────────────────────────────────
def main(evo=None):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, max_episode_steps=150)

    # Infer obs and action dims from the environment
    obs_dict, _ = env.reset()
    obs_dim = len(obs_dict["observation"]) + len(obs_dict["desired_goal"] - obs_dict["achieved_goal"])
    act_dim = env.action_space.shape[0]

    # Load observation limits
    range_vals, min_vals = load_limits()

    n_workers = 50

    pool = Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(
            obs_dim,
            act_dim,
            range_vals,
            min_vals,
        )
    )

    controller = MLPController(obs_dim, act_dim)
    print(f"Controller params: {controller.n_params}")
    print(f"Obs dim: {obs_dim} | Act dim: {act_dim}")

    if os.path.exists(f"../evolution_checkpoints/checkpoint_{evo}.pkl"):
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

        fitnesses = []
        data = []

        for i, (f, success_rate, avg_ep_duration, avg_ep_duration_all_episodes, evaluation_time) in enumerate(results):
            fitnesses.append(f)

            data.append({
                "generation": generation,
                "solution_id": i,
                "fitness": -f,
                "success_rate": success_rate,
                "avg_ep_duration": avg_ep_duration,
                "avg_ep_duration_all_episodes": avg_ep_duration_all_episodes,
                "evaluation_time": evaluation_time
            })

            if f > best_fitness:
                best_fitness = f
                best_params = solutions[i].copy()

        # CMA-ES minimises → negate fitness
        es.tell(solutions, [-f for f in fitnesses])
        es.logger.add()

        generation += 1
        mean_f = -np.mean(fitnesses)

        with open(f"../evolutionary_history/30_total_episodes_10_episodes_fitness_cma_{evo}.jsonl", "a") as f:
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
    np.save(f"../cma_models/best_controller_{evo}.npy", best_params)
    print(f"\nDone. Best fitness: {best_fitness:.3f}")
    print(f"Saved weights to models/best_controller_{evo}.npy")


    del es
    del pool
    del env

    return best_params


if __name__ == "__main__":
    for i in range(5):
        main(evo=i)
        gc.collect()