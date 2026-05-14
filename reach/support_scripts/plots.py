import json
import matplotlib.pyplot as plt

def load_data(jsonl_path):
    generations = []
    avg_success = []
    best_success = []
    avg_final_rewards = []
    best_final_rewards = []
    avg_total_rewards = []
    best_total_rewards = []

    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())

            # Each line has one generation
            gen_key = list(data.keys())[0]
            genomes = data[gen_key]

            success_rates = [g["success_rate"] for g in genomes.values()]
            final_rewards = [g["final_rewards"] for g in genomes.values()]
            total_rewards = [g["total_rewards"] for g in genomes.values()]

            generations.append(gen_key)

            avg_success.append(sum(success_rates) / len(success_rates))
            best_success.append(max(success_rates))

            avg_final_rewards.append(sum(final_rewards) / len(final_rewards))
            best_final_rewards.append(max(final_rewards))

            avg_total_rewards.append(sum(total_rewards) / len(total_rewards))
            best_total_rewards.append(max(total_rewards))

    return generations, avg_success, best_success, avg_final_rewards, best_final_rewards, avg_total_rewards, best_total_rewards


def plot_metrics(jsonl_path):
    gens, avg_s, best_s, avg_f, best_f, avg_t, best_t = load_data(jsonl_path)

    # Convert generation labels to numeric (gen_1 -> 1)
    x = [int(g.split("_")[1]) for g in gens]

    # --- Plot Success Rate ---
    plt.figure()
    plt.plot(x, avg_s, label="Average Success Rate")
    plt.plot(x, best_s, label="Best Success Rate")
    plt.xlabel("Generation")
    plt.ylabel("Success Rate")
    plt.title("Success Rate over Generations")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("success_rate.png")

    # --- Plot Fitness ---
    plt.figure()
    plt.plot(x, avg_f, label="Average Final Rewards")
    plt.plot(x, best_f, label="Best Final Rewards")
    plt.xlabel("Generation")
    plt.ylabel("Final Rewards")
    plt.title("Final Rewards over Generations")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("final_rewards.png")

    
    # --- Plot Total Rewards ---
    plt.figure()
    plt.plot(x, avg_t, label="Average Total Rewards")
    plt.plot(x, best_t, label="Best Total Rewards")
    plt.xlabel("Generation")
    plt.ylabel("Total Rewards")
    plt.title("Total Rewards over Generations")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("total_rewards.png")
    
    

# Run
plot_metrics("dataset.json")