import numpy as np
import gymnasium as gym
import gymnasium_robotics
import cma
import json
import numpy as np

# ── Configuration ──────────────────────────────────────────────
ENV_ID        = "FetchReachDense-v4"
N_EPISODES    = 10       # evaluations per candidate
POP_SIZE      = 100       # CMA-ES lambda (population size)
MAX_GENS      = 500      # generations budget
SIGMA0        = 0.5      # initial step size
ALPHA         = 0.3      # distance penalty weight in fitness
HIDDEN        = 32       # hidden units in the controller MLP

# ── Controller: small MLP obs → action ─────────────────────────
class MLPController:
    def __init__(self, obs_dim, act_dim, hidden=HIDDEN):
        self.shapes = [
            (hidden, obs_dim), (hidden,),
            (hidden, hidden), (hidden,),
            (act_dim, hidden), (act_dim,),
        ]
        self.n_params = sum(np.prod(s) for s in self.shapes)

    def set_weights(self, params):
        self.params = params

    def act(self, obs):
        x = obs.astype(np.float32)
        idx = 0
        layers = []
        for i, shape in enumerate(self.shapes):
            size = np.prod(shape)
            layers.append(self.params[idx:idx+size].reshape(shape))
            idx += size
        # W1, b1, W2, b2, W3, b3
        W1,b1,W2,b2,W3,b3 = layers
        x = np.tanh(W1 @ x + b1)
        x = np.tanh(W2 @ x + b2)
        x = np.tanh(W3 @ x + b3)   # actions clipped to [-1,1] by tanh
        return x


# ── Fitness function ────────────────────────────────────────────
def evaluate(params, env, controller, range_vals, min_vals, n_episodes=N_EPISODES):
    """
    Returns scalar fitness (higher = better).
    fitness = success_rate - ALPHA * mean_goal_distance
    """
    controller.set_weights(params)
    successes = 0
    distances = []
    total_rewards = []

    for _ in range(n_episodes):
        obs_dict, _ = env.reset()
        obs = np.concatenate([
            obs_dict["observation"],
            obs_dict["desired_goal"] - obs_dict["achieved_goal"]])
        done = False
        ep_dist = []
        total_reward = 0

        for _ in range(env._max_episode_steps):
            obs = normalize_observation(obs, range_vals, min_vals)

            action = controller.act(obs)
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = np.concatenate([
                obs_dict["observation"],
                obs_dict["desired_goal"] - obs_dict["achieved_goal"]])
            # Distance to goal (Euclidean, from achieved vs desired)
            dist = np.linalg.norm(
                obs_dict["achieved_goal"] - obs_dict["desired_goal"]
            )
            ep_dist.append(dist)
            total_reward += reward

        if info.get("is_success", False):
            successes += 1
        distances.append(np.mean(ep_dist))
        total_rewards.append(total_reward)

    success_rate = successes / n_episodes
    mean_dist    = np.mean(distances)
    fitness      = np.mean(total_rewards) * 10#success_rate - ALPHA * mean_dist
    return fitness, success_rate   # CMA-ES minimises, so we negate below

def load_limits(path="observation_limits_13_obs.json"):
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

# ── Main evolution loop ─────────────────────────────────────────
def main():
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, max_episode_steps=100)

    # Infer obs and action dims from the environment
    obs_dict, _ = env.reset()
    obs_dim = len(obs_dict["observation"]) + len(obs_dict["desired_goal"] - obs_dict["achieved_goal"])
    act_dim = env.action_space.shape[0]

    # Load observation limits
    range_vals, min_vals = load_limits()

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

    x0 = np.zeros(controller.n_params)
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, opts)

    best_fitness = -np.inf
    best_params  = None

    generation = 0
    while not es.stop():
        solutions = es.ask()          # sample λ candidates
        fitnesses = []
        data = []
        i = 0

        for params in solutions:
            f, success_rate = evaluate(params, env, controller, range_vals, min_vals)
            data.append({
                "generation": generation,
                "solution_id": i,
                "fitness": f,
                "success_rate": success_rate,
            })
            fitnesses.append(f)
            if f > best_fitness:
                best_fitness = f
                best_params  = params.copy()
            
            i += 1

        # CMA-ES minimises → negate fitness
        es.tell(solutions, [-f for f in fitnesses])
        es.logger.add()

        generation += 1
        mean_f = np.mean(fitnesses)

        with open("cma_results.jsonl", "a") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"Gen {generation:4d} | best={best_fitness:.3f} | mean={mean_f:.3f} | sigma={es.sigma:.4f}")

    env.close()

    # Save best controller
    np.save("best_controller.npy", best_params)
    print(f"\nDone. Best fitness: {best_fitness:.3f}")
    print("Saved weights to best_controller.npy")

    return best_params


if __name__ == "__main__":
    main()