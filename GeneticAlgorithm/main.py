# from algorithm.algorithm import random_search

# if __name__ == "__main__":
#     best = random_search("benchmarks/matmul.cpp", iterations=30)
#     print("BEST:", best)

import time
from algorithm.geneticAlgorithm import run_genetic_algorithm
from algorithm.test_harness import translate_chromosome_to_flags, get_baseline_runtime
from algorithm.flags import CORE_FLAGS

# --- EXPERIMENT CONFIGURATION ---

# DUMMY CORE FLAG LIST (Replace this with your actual 15-20 curated flag names)
# These names MUST NOT include the -f or -fno- prefix!

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    print("\n=============================================")
    print("      COMPILER FLAG OPTIMIZATION STUDY")
    print("=============================================")
    print(f"Using {len(CORE_FLAGS)} boolean flags for search space size 2^{len(CORE_FLAGS)}")
    
    start_global_time = time.time()
    
    # --- 1. ESTABLISH BASELINE (-O3 ONLY) ---
    # This calls the new function in test_harness.py
    baseline_runtime = get_baseline_runtime()
    
    if baseline_runtime == float("inf"):
        print("Cannot proceed: Baseline calculation failed. Check compilation errors in test_harness logs.")
    else:
        # --- 2. RUN THE GENETIC ALGORITHM ---
        print("\n--- Starting Genetic Algorithm Search ---")
        try:
            results = run_genetic_algorithm(CORE_FLAGS)
        except Exception as e:
            print(f"\nFATAL ERROR during GA execution: {e}")
            results = None

        end_global_time = time.time()
        
        if results and results['best_fitness'] != float("inf"):
            # --- 3. REPORT FINAL RESULTS ---
            
            ga_runtime = results['best_fitness']
            # Calculate speedup percentage: (Baseline - Optimal) / Baseline * 100
            speedup_percentage = ((baseline_runtime - ga_runtime) / baseline_runtime) * 100

            print("\n=============================================")
            print("       GENETIC ALGORITHM OPTIMIZATION REPORT")
            print("=============================================")
            print(f"Total Experiment Time: {end_global_time - start_global_time:.2f} seconds")
            
            print("\n--- Performance Comparison ---")
            print(f"1. Baseline Runtime (-O3): {baseline_runtime:.4f} seconds")
            print(f"2. GA Optimal Runtime:    {ga_runtime:.4f} seconds")
            print(f"** Speedup Achieved:   {speedup_percentage:.2f}% **")
            
            # Translate the best chromosome (0s/1s list) back into human-readable flags
            final_flags = translate_chromosome_to_flags(results['best_chromosome'], CORE_FLAGS)
            
            print("\nOptimal Flag Set:")
            print(f"g++-15 {' '.join(final_flags)}")
            print("=============================================")
        else:
            print("\n=============================================")
            print("       EXPERIMENT FAILED OR DID NOT CONVERGE")
            print("=============================================")