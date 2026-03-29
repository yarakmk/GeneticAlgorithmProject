import time
from algorithm.geneticAlgorithm import run_genetic_algorithm
from algorithm.test_harness import translate_chromosome_to_flags, get_baseline_runtime

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

POLYBENCH_ROOT = 'PolyBenchC-4.2.1'
UTILITIES      = f'{POLYBENCH_ROOT}/utilities'
POLYBENCH_C    = f'{UTILITIES}/polybench.c'
DEFINES        = ['POLYBENCH_TIME', 'STANDARD_DATASET']

BENCHMARKS = [
    {
        'name'    : 'gemm',
        'path'    : f'{POLYBENCH_ROOT}/linear-algebra/blas/gemm/gemm.c',
        'includes': [UTILITIES, f'{POLYBENCH_ROOT}/linear-algebra/blas/gemm'],
    },
    {
        'name'    : '2mm',
        'path'    : f'{POLYBENCH_ROOT}/linear-algebra/kernels/2mm/2mm.c',
        'includes': [UTILITIES, f'{POLYBENCH_ROOT}/linear-algebra/kernels/2mm'],
    },
    {
        'name'    : 'nussinov',
        'path'    : f'{POLYBENCH_ROOT}/medley/nussinov/nussinov.c',
        'includes': [UTILITIES, f'{POLYBENCH_ROOT}/medley/nussinov'],
    },
    {
        'name'    : 'jacobi-2d',
        'path'    : f'{POLYBENCH_ROOT}/stencils/jacobi-2d/jacobi-2d.c',
        'includes': [UTILITIES, f'{POLYBENCH_ROOT}/stencils/jacobi-2d'],
    },
    {
        'name'    : 'ludcmp',
        'path'    : f'{POLYBENCH_ROOT}/linear-algebra/solvers/ludcmp/ludcmp.c',
        'includes': [UTILITIES, f'{POLYBENCH_ROOT}/linear-algebra/solvers/ludcmp'],
    },
]

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print("\n=============================================")
    print("      SAGE: COMPILER FLAG OPTIMIZATION")
    print("=============================================")

    all_results = []
    start_global_time = time.time()

    for benchmark in BENCHMARKS:

        print(f"\n{'='*60}")
        print(f"  Benchmark : {benchmark['name']}")
        print(f"  Path      : {benchmark['path']}")
        print(f"{'='*60}")

        # --- 1. Establish -O3 baseline ---
        baseline = get_baseline_runtime(
            benchmark_path  = benchmark['path'],
            polybench       = True,
            extra_sources   = [POLYBENCH_C],
            extra_includes  = benchmark['includes'],
            defines         = DEFINES
        )

        if baseline == float('inf'):
            print(f"  ERROR: Baseline failed for {benchmark['name']} — skipping.")
            continue

        # --- 2. Run SAGE ---
        try:
            result = run_genetic_algorithm(
                benchmark_path  = benchmark['path'],
                polybench       = True,
                extra_sources   = [POLYBENCH_C],
                extra_includes  = benchmark['includes'],
                defines         = DEFINES
            )
        except Exception as e:
            print(f"  FATAL ERROR during GA on {benchmark['name']}: {e}")
            continue

        # --- 3. Report results for this benchmark ---
        if result and result['best_fitness'] != float('inf'):
            ga_runtime = result['best_fitness']
            speedup    = ((baseline - ga_runtime) / baseline) * 100

            print(f"\n  --- Results: {benchmark['name']} ---")
            print(f"  Baseline (-O3)  : {baseline:.4f}s")
            print(f"  SAGE optimal    : {ga_runtime:.4f}s")
            print(f"  Speedup         : {speedup:.2f}%")

            all_results.append({
                'name'    : benchmark['name'],
                'baseline': baseline,
                'ga'      : ga_runtime,
                'speedup' : speedup,
                'flags'   : result.get('best_flags', [])
            })
        else:
            print(f"  GA did not converge for {benchmark['name']}.")

    end_global_time = time.time()

    # --- 4. Summary table ---
    if all_results:
        print(f"\n{'='*60}")
        print(f"  FINAL SUMMARY")
        print(f"  Total experiment time: {end_global_time - start_global_time:.2f}s")
        print(f"{'='*60}")
        print(f"  {'Benchmark':<15} {'Baseline':>10} {'GA':>10} {'Speedup':>10}")
        print(f"  {'-'*47}")
        for r in all_results:
            print(f"  {r['name']:<15} {r['baseline']:>9.4f}s {r['ga']:>9.4f}s {r['speedup']:>9.2f}%")
        print(f"{'='*60}")
