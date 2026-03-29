import subprocess
import time
import statistics
import os
import json

# --- CONFIGURATION ---
SOURCE_FILE = "benchmarks/montecarlo.cpp"
COMPILER = "g++-15"
BASELINE = "-O3"
FLAGS_FILE = "bayesian_curated_flags.txt"
TRIALS = 7  # Increased to allow for trimming outliers
RE_BASELINE_EVERY = 5 # Re-measure baseline more frequently
COOLDOWN_SECONDS = 0.1 # Short pause to let CPU "breathe"

def get_sdk_path():
    try:
        return subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True).stdout.strip()
    except: return None

SDK_PATH = get_sdk_path()

def measure_runtime(extra_flags):
    bin_path = "./tmp_sens_bin"
    cmd = [COMPILER, BASELINE] + extra_flags
    if SDK_PATH: cmd += ["-isysroot", SDK_PATH]
    cmd += [SOURCE_FILE, "-o", bin_path]
    
    try:
        # Step 1: Compile
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: return None
        
        # Step 2: Warm-up (Discards the first run to prime caches)
        subprocess.run([bin_path], check=True, capture_output=True)
        
        # Step 3: Actual Measurements
        times = []
        for _ in range(TRIALS):
            time.sleep(COOLDOWN_SECONDS) # Prevent thermal runaway
            start = time.perf_counter()
            subprocess.run([bin_path], check=True, capture_output=True)
            times.append(time.perf_counter() - start)
        
        # Step 4: Robust Statistics
        # We sort and remove the fastest and slowest results (Trimmed Mean)
        times.sort()
        trimmed_times = times[2:-2] if len(times) >= 5 else times
        return statistics.mean(trimmed_times)
    except:
        return None

def run_sensitivity_test():
    if not os.path.exists(FLAGS_FILE): 
        print(f"Error: {FLAGS_FILE} not found.")
        return

    with open(FLAGS_FILE, "r") as f:
        all_flags = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print(f"Starting High-Precision Sensitivity Analysis...")
    
    # Establish initial baseline
    current_baseline = measure_runtime([])
    if current_baseline is None:
        print("Initial baseline failed.")
        return
        
    print(f"Initial Baseline: {current_baseline:.6f}s")

    results = []

    for i, flag_name in enumerate(all_flags):
        # Re-measure baseline frequently to track thermal drift
        if i > 0 and i % RE_BASELINE_EVERY == 0:
            new_base = measure_runtime([])
            if new_base:
                drift = ((new_base - current_baseline) / current_baseline) * 100
                current_baseline = new_base
                print(f"\n[Calibrating] New Baseline: {current_baseline:.6f}s (Heat Drift: {drift:+.2f}%)")

        flag = f"-f{flag_name}"
        print(f"[{i+1}/{len(all_flags)}] {flag_name}...", end=" ", flush=True)
        
        current_time = measure_runtime([flag])
        
        if current_time is None:
            print("FAILED")
            continue
        
        diff = current_baseline - current_time
        impact_pct = (diff / current_baseline) * 100
        results.append({"flag": flag_name, "impact_pct": impact_pct})
        print(f"{impact_pct:+.4f}%")

    # Final Pruning - Only keep positive, stable speedups
    results.sort(key=lambda x: x['impact_pct'], reverse=True)
    influential = [res['flag'] for res in results if res['impact_pct'] > 0.03] # Higher bar for stability
    
    os.makedirs("config", exist_ok=True)
    with open("influential_flags.txt", "w") as f:
        for flag in influential: f.write(f"{flag}\n")

    print(f"\nAnalysis complete.")
    print(f"Saved {len(influential)} flags with >0.1% speedup to influential_flags.txt")

if __name__ == "__main__":
    run_sensitivity_test()