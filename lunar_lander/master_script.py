import os
import subprocess
import sys

# ---------------- CONFIG ----------------
NUM_RUNS = 3  # how many times to repeat
H5_PATH = "data/episodes.h5"
DB_SCRIPT = "database_builder.py"
TRAIN_SCRIPT = "train_tcn.py"

# Arguments for database_builder.py
DB_ARGS = ["--evolutions", "1", "--generations", "1", "--episodes", "10"]

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# ---------------- MASTER LOOP ----------------
for run_idx in range(1, NUM_RUNS + 1):
    print(f"\n===== RUN {run_idx}/{NUM_RUNS} =====\n")

    # Run database_builder.py with arguments
    print(f"Running {DB_SCRIPT} with args: {DB_ARGS} ...")
    subprocess.run([sys.executable, DB_SCRIPT] + DB_ARGS, check=True)
    print("Database generation complete.")

    #  Run train_tcn.py
    print(f"Running {TRAIN_SCRIPT} ...")
    subprocess.run([sys.executable, TRAIN_SCRIPT], check=True)
    print("Training complete. Metrics logged.")

     # Delete episodes.h5
    if os.path.exists(H5_PATH):
        os.remove(H5_PATH)
        print(f"Deleted {H5_PATH}")