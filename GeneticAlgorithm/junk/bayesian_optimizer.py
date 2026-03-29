import subprocess
import time
import statistics
import os
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

# --- CONFIGURATION ---
SOURCE_FILE = "benchmarks/monte.cpp"
COMPILER = "g++-15"
BASELINE_O3 = "-O3"
FLAGS_FILE = "all_flags.txt" # Use your curated list
TRIALS = 3
N_INITIAL = 15
N_CALLS = 60  # How many different combinations to test

def get_sdk_path():
    try:
        return subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True).stdout.strip()
    except: return None

SDK_PATH = get_sdk_path()

# 1. Load the flags you want to optimize
with open(FLAGS_FILE, "r") as f:
    all_flags = [line.strip() for line in f if line.strip()]

# 2. Define the Search Space: Each flag is either 0 (off) or 1 (on)
# We use Categorical([0, 1]) for each flag
search_space = [Categorical([0, 1], name=flag) for flag in all_flags]

def measure_runtime(extra_flags):
    bin_path = "./tmp_bayesian_bin"
    cmd = [COMPILER, BASELINE_O3] + extra_flags
    if SDK_PATH: cmd += ["-isysroot", SDK_PATH]
    cmd += [SOURCE_FILE, "-o", bin_path]
    
    try:
        # Compile
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: return 999.0 # Penalty for failed compilation
        
        # Run multiple trials
        times = []
        for _ in range(TRIALS):
            start = time.perf_counter()
            subprocess.run([bin_path], check=True, capture_output=True)
            times.append(time.perf_counter() - start)
        return statistics.median(times)
    except:
        return 999.0

# 3. The Objective Function that Scikit-Optimize will minimize
@use_named_args(search_space)
def objective(**flags_config):
    # Convert the dict of 0/1s into a list of -f strings
    active_flags = [f"-f{name}" for name, value in flags_config.items() if value == 1]
    
    print(f"\nTesting {len(active_flags)} flags...", end=" ")
    runtime = measure_runtime(active_flags)
    print(f"Result: {runtime:.6f}s")
    
    return runtime

def run_bayesian_opt():
    print(f"Starting Bayesian Optimization with {len(all_flags)} flags...")
    print(f"Total iterations planned: {N_CALLS}")
    
    # 4. Run the Optimizer
    # acq_func='EI' means 'Expected Improvement' (balances exploration/exploitation)
    res = gp_minimize(
        objective, 
        search_space, 
        n_calls=N_CALLS, 
        n_initial_points= N_INITIAL, # Start with n_initial random samples to build the initial map
        random_state=42,
        acq_func='EI'
    )
    
    # 5. Output Results
    print("\n" + "="*30)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Runtime Found: {res.fun:.6f}s")
    
    # Identify the best configuration
    best_config = []
    for i, val in enumerate(res.x):
        if val == 1:
            best_config.append(all_flags[i])
            
    print(f"Best Flag Combination ({len(best_config)} flags):")
    for flag in best_config:
        print(f"  -f{flag}")

    # Save the result so you can use it as a 'curated' list for the GA later
    with open("bayesian_curated_flags.txt", "w") as f:
        for flag in best_config:
            f.write(f"{flag}\n")
            
    print("Saved best combination to: bayesian_curated_flags.txt")
if __name__ == "__main__":
    # Ensure scikit-optimize is installed: pip install scikit-optimize
    try:
        run_bayesian_opt()
    except ImportError:
        print("Please install scikit-optimize: pip install scikit-optimize")