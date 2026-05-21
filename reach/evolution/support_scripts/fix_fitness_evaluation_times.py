import json
from pathlib import Path
import glob

# folder containing the files
DATA_DIR = Path("../evolutionary_history/30_total_episodes_10_episodes_fitness_evolution")

for file_path in DATA_DIR.glob("*.json*"):

    print(f"Processing {file_path.name}...")
    filename = file_path.name

    n = 10

    new_lines = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            current = data["evaluation_time"]
            data["evaluation_time"] = current / 30 * n

            new_lines.append(json.dumps(data))

    # overwrite file
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"Updated {filename} with n={n}")