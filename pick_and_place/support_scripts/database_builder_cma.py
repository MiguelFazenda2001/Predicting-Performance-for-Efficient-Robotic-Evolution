import numpy as np
import gymnasium as gym
import gymnasium_robotics
import cma
import json
import numpy as np
from multiprocessing import Pool
import h5py
import gc

"""

Normalize distances to [0,1] per episode

"""


# ── Configuration ──────────────────────────────────────────────
ENV_ID        = "FetchPushDense-v4"
N_EPISODES    = 10       # evaluations per candidate
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
    return evaluate(
        params,
        _worker_env,
        _worker_controller,
        _range_vals,
        _min_vals,
    )

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
def evaluate(params, env, controller, range_vals, min_vals, n_episodes=N_EPISODES):
    """
    Returns scalar fitness (higher = better).
    fitness = success_rate - ALPHA * mean_goal_distance
    """
    controller.set_weights(params)
    successes = 0
    total_rewards = []
    final_rewards = []
    durations = []

    genome_data = {
        "genome_id": id(params),
        "episodes_data": [],
        "success_rate": 0,
        "avg_duration": 0,
        "fitness": 0
    }

    for _ in range(n_episodes):
        episode_data = {
            "observations": [],
            "actions": [],
            "success": 0,
            "last_reward": 0,
            "duration": 0
        }
        observations, actions = [], []
        obs_dict, _ = env.reset()
        obs = np.concatenate([
            obs_dict["observation"],
            obs_dict["desired_goal"] - obs_dict["achieved_goal"]])
        done = False

        total_reward = 0
        success_step = None

        for step in range(env._max_episode_steps):
            obs = normalize_observation(obs, range_vals, min_vals)

            observations.append(obs)
            action = controller.act(obs)
            actions.append(action)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = np.concatenate([
                obs_dict["observation"],
                obs_dict["desired_goal"] - obs_dict["achieved_goal"]])

            total_reward += reward

            if reward >= -0.05 and success_step is None:
                success_step = step
            elif reward < -0.05:
                success_step = None

            if terminated or truncated:
                break

        if reward > -0.05:  
            successes += 1
            success = 1
        else:
            success = 0

        if success_step is None:
            success_step = env._max_episode_steps - 1

        total_rewards.append(total_reward)
        final_rewards.append(reward)

        duration = success_step / env.metadata["render_fps"]
        durations.append(duration)

        episode_data["observations"] = observations
        episode_data["actions"] = actions
        episode_data["success"] = success
        episode_data["duration"] = duration
        episode_data["last_reward"] = reward
        genome_data["episodes_data"].append(episode_data)
    
    success_rate = successes / n_episodes
    fitness = np.mean(final_rewards) * 100 + success_rate * 100  # higher success and lower distance → higher fitness

    genome_data["success_rate"] = success_rate
    genome_data["avg_duration"] = float(np.mean(durations))
    genome_data["fitness"] = float(fitness)

    env.close()

    return fitness, success_rate, genome_data   # CMA-ES minimises, so we negate below

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

def save_population_data_to_h5(evo, h5_path, current_generation, population_data):
    with h5py.File(h5_path, "a") as h5file:
        for genome_data in population_data:
            random_number = np.random.rand()  # Generate a random number for uniqueness
            if genome_data["success_rate"] < 0.1 and random_number > 0.005:
                continue
            elif  genome_data["success_rate"] < 0.2 and random_number > 0.005:
                continue
            elif  genome_data["success_rate"] < 0.3 and random_number > 0.01:
                continue
            elif genome_data["success_rate"] < 0.4 and random_number > 0.1:
                continue
            elif genome_data["success_rate"] < 0.5 and random_number > 0.5:
                continue
            else:    
                group_name = f"evo_{evo}/gen_{current_generation}/genome_{genome_data['genome_id']}"
                if group_name in h5file:
                    del h5file[group_name]  # Remove existing group to avoid duplication
                group = h5file.create_group(group_name)
                group.attrs["success_rate"] = genome_data["success_rate"]
                group.attrs["avg_duration"] = genome_data["avg_duration"]
                group.attrs["fitness"] = genome_data["fitness"]

                for episode_id, episode_data in enumerate(genome_data["episodes_data"]):
                    episode_group = group.create_group(f"episode_{episode_id}")
                    episode_group.create_dataset("observations", data=np.array(episode_data["observations"]),compression="gzip",
                    compression_opts=4)
                    episode_group.create_dataset("actions", data=np.array(episode_data["actions"]),compression="gzip",
                    compression_opts=4)
                    episode_group.attrs["success"] = episode_data["success"]
                    episode_group.attrs["duration"] = episode_data["duration"]
                    episode_group.attrs["last_reward"] = episode_data["last_reward"]
                h5file.flush()



# ── Main evolution loop ─────────────────────────────────────────
def main(evo=None, h5_path=None):
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

    # CMA-ES options
    opts = cma.CMAOptions()
    opts["popsize"]    = POP_SIZE
    opts["maxiter"]    = MAX_GENS
    opts["verbose"]    = 1
    opts["tolx"]       = 1e-5
    opts["tolfun"]     = 1e-5
    opts["CMA_elitist"] = True

    x0 = np.zeros(controller.n_params)
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)
    stagnation_counter = 0

    best_fitness = -np.inf
    best_params  = None

    generation = 0
    while not es.stop():
        solutions = es.ask()          # sample λ candidates
        fitnesses = []
        data = []
        i = 0

        results = pool.map(evaluate_worker, solutions, chunksize=10)

        fitnesses = []
        data = []
        population_data = []

        for i, (f, success_rate, genome_data) in enumerate(results):
            fitnesses.append(f)

            data.append({
                "generation": generation,
                "solution_id": i,
                "fitness": -f,
                "success_rate": success_rate,
            })
            population_data.append(genome_data)

            if f > best_fitness:
                best_fitness = f
                best_params = solutions[i].copy()

        # CMA-ES minimises → negate fitness
        es.tell(solutions, [-f for f in fitnesses])
        es.logger.add()

        generation += 1
        mean_f = -np.mean(fitnesses)

        with open(f"history/cma_results_{evo}.jsonl", "a") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        
        print(f"Gen {generation:4d} | best={best_fitness:.3f} | mean={mean_f:.3f} | sigma={es.sigma:.4f}")
        save_population_data_to_h5(evo, h5_path, generation, population_data)

    pool.close()
    pool.join()
    pool.terminate()

    env.close()

    # Save best controller
    np.save(f"models/best_controller_{evo}.npy", best_params)
    print(f"\nDone. Best fitness: {best_fitness:.3f}")
    print(f"Saved weights to models/best_controller_{evo}.npy")


    del es
    del pool
    del env

    return best_params


if __name__ == "__main__":
    for i in range(20):
        main(evo=i+11, h5_path=f"/mnt/DATA/miguelfazenda/pickplace/dataset/raw/train_episodes_{i+11}.h5")
        gc.collect()