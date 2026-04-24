import time
from algorithm.geneticAlgorithm import run_genetic_algorithm
from algorithm.test_harness import translate_chromosome_to_flags, get_baseline_runtime
import argparse
import json
import os
import algorithm.geneticAlgorithm as ga
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAGE: Compiler flag optimisation"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        help="Path to a single benchmark (overrides default suite)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file"
    )

    parser.add_argument("--population-size", type=int)
    parser.add_argument("--crossover-type", type=str)
    parser.add_argument("--mutation-type", type=str)
    parser.add_argument("--selection-type", type=str)
    parser.add_argument("--crossover-rate", type=float)
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--elitism-ratio", type=float)
    parser.add_argument("--parents-portion", type=float)
    parser.add_argument("--max-generations", type=int)
    parser.add_argument("--max-no-improvement", type=int)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: None = non-deterministic)")

    return parser.parse_args()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    args = parse_args()

    # --- Load JSON config ---
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)

    # --- Apply config to GA module ---
    ga.POPULATION_SIZE = config.get("population_size", ga.POPULATION_SIZE)
    ga.CROSSOVER_TYPE  = config.get("crossover_type", ga.CROSSOVER_TYPE)
    ga.MUTATION_TYPE   = config.get("mutation_type", ga.MUTATION_TYPE)
    ga.SELECTION_TYPE  = config.get("selection_type", ga.SELECTION_TYPE)

    # --- CLI overrides (take precedence) ---
    if args.population_size is not None:
        ga.POPULATION_SIZE = args.population_size

    if args.crossover_type is not None:
        ga.CROSSOVER_TYPE = args.crossover_type

    if args.mutation_type is not None:
        ga.MUTATION_TYPE = args.mutation_type

    if args.selection_type is not None:
        ga.SELECTION_TYPE = args.selection_type

    if args.crossover_rate is not None:
        ga.CROSSOVER_RATE = args.crossover_rate

    if args.mutation_rate is not None:
        ga.MUTATION_RATE = args.mutation_rate

    if args.elitism_ratio is not None:
        ga.ELITISM_RATIO = args.elitism_ratio

    if args.parents_portion is not None:
        ga.PARENTS_PORTION = args.parents_portion

    if args.max_generations is not None:
        ga.MAX_GENERATIONS = args.max_generations

    if args.max_no_improvement is not None:
        ga.MAX_NO_IMPROVEMENT = args.max_no_improvement

    seed = config.get("random_seed", None)
    if args.seed is not None:
        seed = args.seed
    ga.RANDOM_SEED = seed

    print("\n=============================================")
    print("      SAGE: COMPILER FLAG OPTIMIZATION")
    print("=============================================")

    all_results = []
    start_global_time = time.time()

    benchmarks_to_run = BENCHMARKS

    if args.benchmark:
        benchmarks_to_run = [{
            "name": os.path.splitext(os.path.basename(args.benchmark))[0],
            "path": args.benchmark,
            "includes": [UTILITIES, os.path.dirname(args.benchmark)]
        }]

    for benchmark in benchmarks_to_run:

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
