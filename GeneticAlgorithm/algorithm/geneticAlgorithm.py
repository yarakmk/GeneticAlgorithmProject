import random
import statistics
import time
import math
from algorithm.test_harness import get_fitness_score, get_baseline_runtime
from algorithm.flag_matcher import build_flag_list

import json
import os
CONFIG_FILE = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

# Load config at module level
_config = load_config()

POPULATION_SIZE    = _config.get('population_size',    64)
CROSSOVER_TYPE     = _config.get('crossover_type',     'one_point')
CROSSOVER_RATE     = _config.get('crossover_rate',     0.4711)
MUTATION_TYPE      = _config.get('mutation_type',      'gauss_by_center')
MUTATION_RATE      = _config.get('mutation_rate',      0.1868)
SELECTION_TYPE     = _config.get('selection_type',     'ranking')
ELITISM_RATIO      = _config.get('elitism_ratio',      0.2998)
PARENTS_PORTION    = _config.get('parents_portion',    0.5527)
MAX_GENERATIONS    = _config.get('max_generations',    33)
MAX_NO_IMPROVEMENT = _config.get('max_no_improvement', 25)
RANDOM_SEED        = _config.get('random_seed',        None)
TOURNAMENT_SIZE = 5            # Only used when SELECTION_TYPE = 'tournament'

CFSCA_FLAGS_FILE = 'CFSCA_flags.txt'

# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

def crossover_one_point(parent_a, parent_b):
    """
    Single-Point Crossover.
    Picks one random point and swaps everything after it between parents.

    Example (split at index 3):
      Parent A: [1, 0, 1 | 0, 0, 1]
      Parent B: [0, 1, 0 | 1, 1, 0]
      Child  A: [1, 0, 1 | 1, 1, 0]
      Child  B: [0, 1, 0 | 0, 0, 1]
    """
    point = random.randint(1, len(parent_a) - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def crossover_two_point(parent_a, parent_b):
    """
    Two-Point Crossover.
    Picks two random points and swaps the segment between them.

    Example (points at 2 and 5):
      Parent A: [1, 0 | 1, 0, 0 | 1]
      Parent B: [0, 1 | 0, 1, 1 | 0]
      Child  A: [1, 0 | 0, 1, 1 | 1]
      Child  B: [0, 1 | 1, 0, 0 | 0]
    """
    point1 = random.randint(1, len(parent_a) - 2)
    point2 = random.randint(point1 + 1, len(parent_a) - 1)
    child_a = parent_a[:point1] + parent_b[point1:point2] + parent_a[point2:]
    child_b = parent_b[:point1] + parent_a[point1:point2] + parent_b[point2:]
    return child_a, child_b


def crossover_uniform(parent_a, parent_b):
    """
    Uniform Crossover.
    Each gene is independently picked from either parent with 50/50 probability.
    Produces more diverse offspring than point-based crossover.

    Example:
      Parent A: [1, 0, 1, 0, 0, 1]
      Parent B: [0, 1, 0, 1, 1, 0]
      Mask:     [A, B, A, A, B, B]  (random per gene)
      Child  A: [1, 1, 1, 0, 1, 0]
      Child  B: [0, 0, 0, 1, 0, 1]
    """
    child_a, child_b = [], []
    for gene_a, gene_b in zip(parent_a, parent_b):
        if random.random() < 0.5:
            child_a.append(gene_a)
            child_b.append(gene_b)
        else:
            child_a.append(gene_b)
            child_b.append(gene_a)
    return child_a, child_b


def crossover_shuffle(parent_a, parent_b):
    """
    Shuffle (Scramble) Crossover.
    Shuffles the parents' genes randomly before doing one-point crossover,
    then unshuffles. This reduces positional bias.
    """
    length = len(parent_a)
    # Create a shuffled index order
    indices = list(range(length))
    random.shuffle(indices)

    # Rearrange both parents according to the shuffled order
    shuffled_a = [parent_a[i] for i in indices]
    shuffled_b = [parent_b[i] for i in indices]

    # One-point crossover on shuffled versions
    shuffled_ca, shuffled_cb = crossover_one_point(shuffled_a, shuffled_b)

    # Unshuffle: put genes back in their original positions
    child_a = [None] * length
    child_b = [None] * length
    for new_pos, orig_pos in enumerate(indices):
        child_a[orig_pos] = shuffled_ca[new_pos]
        child_b[orig_pos] = shuffled_cb[new_pos]

    return child_a, child_b


def crossover_segment(parent_a, parent_b):
    """
    Segment-Based Crossover.
    Picks multiple random segments from each parent alternately to form children.
    Provides higher gene diversity than standard point-based crossover.

    Example (3 segments):
      Parent A: [1, 0, 1, 0, 0, 1, 1, 0]
      Parent B: [0, 1, 0, 1, 1, 0, 0, 1]
      Segments: [--A--][--B--][---A---]
      Child  A: [1, 0, 0, 1, 0, 1, 1, 0]
    """
    length = len(parent_a)
    # Pick 2 to 4 random cut points to define segments
    num_cuts = random.randint(2, min(4, length - 1))
    cut_points = sorted(random.sample(range(1, length), num_cuts))
    cut_points = [0] + cut_points + [length]

    child_a, child_b = [], []
    # Alternate which parent each segment comes from
    for seg_idx in range(len(cut_points) - 1):
        start, end = cut_points[seg_idx], cut_points[seg_idx + 1]
        if seg_idx % 2 == 0:
            child_a.extend(parent_a[start:end])
            child_b.extend(parent_b[start:end])
        else:
            child_a.extend(parent_b[start:end])
            child_b.extend(parent_a[start:end])

    return child_a, child_b


def crossover(parent_a, parent_b):
    """
    Dispatcher: applies crossover only if random roll < CROSSOVER_RATE,
    then delegates to the selected crossover type.
    """
    if random.random() >= CROSSOVER_RATE:
        # No crossover: return copies of parents unchanged
        return parent_a[:], parent_b[:]

    if CROSSOVER_TYPE == 'one_point':
        return crossover_one_point(parent_a, parent_b)
    elif CROSSOVER_TYPE == 'two_point':
        return crossover_two_point(parent_a, parent_b)
    elif CROSSOVER_TYPE == 'uniform':
        return crossover_uniform(parent_a, parent_b)
    elif CROSSOVER_TYPE == 'shuffle':
        return crossover_shuffle(parent_a, parent_b)
    elif CROSSOVER_TYPE == 'segment':
        return crossover_segment(parent_a, parent_b)
    else:
        raise ValueError(f"Unknown crossover type: '{CROSSOVER_TYPE}'")


# =============================================================================
# MUTATION OPERATORS
# =============================================================================

def mutate_bit_flip(chromosome):
    """
    Bit-Flip Mutation.
    Each gene independently has a MUTATION_RATE chance of flipping 0->1 or 1->0.
    Simple and effective for binary chromosomes.
    """
    return [1 - gene if random.random() < MUTATION_RATE else gene
            for gene in chromosome]


def mutate_gauss_by_center(chromosome):
    """
    Gauss-by-Center Mutation.
    Adds a value drawn from a normal distribution (mean=0) to each gene,
    then rounds back to binary. Genes near 0 or 1 are less likely to flip
    than bit-flip mutation — it's a 'softer' mutation that preserves
    strong genes more often.

    The standard deviation is scaled by MUTATION_RATE so the tuner
    can still control the intensity via the same parameter.
    """
    mutated = []
    for gene in chromosome:
        # Draw from normal distribution centred at current gene value
        noisy_value = gene + random.gauss(0, MUTATION_RATE)
        # Clamp to [0, 1] and round to binary
        mutated.append(round(max(0.0, min(1.0, noisy_value))))
    return mutated


def mutate_uniform(chromosome):
    """
    Uniform Mutation.
    Each gene that is selected for mutation (probability = MUTATION_RATE)
    is replaced with a completely random binary value (not necessarily a flip).
    For binary chromosomes this behaves similarly to bit-flip, but the
    distinction matters more for real-valued encodings.
    """
    return [random.randint(0, 1) if random.random() < MUTATION_RATE else gene
            for gene in chromosome]


def mutate(chromosome):
    """
    Dispatcher: delegates to the selected mutation type.
    """
    if MUTATION_TYPE == 'bit_flip':
        return mutate_bit_flip(chromosome)
    elif MUTATION_TYPE == 'gauss_by_center':
        return mutate_gauss_by_center(chromosome)
    elif MUTATION_TYPE == 'uniform_mutation':
        return mutate_uniform(chromosome)
    else:
        raise ValueError(f"Unknown mutation type: '{MUTATION_TYPE}'")


# =============================================================================
# SELECTION OPERATORS
# =============================================================================

def selection_fully_random(population, fitness_scores, n):
    """
    Fully Random Selection.
    Selects n parents completely at random — ignores fitness entirely.
    Useful as a baseline to confirm that fitness-guided selection matters.
    """
    return random.choices(population, k=n)


def selection_roulette(population, fitness_scores, n):
    """
    Roulette Wheel (Fitness-Proportionate) Selection.
    Each individual's chance of being selected is proportional to its fitness.
    Since we MINIMISE (lower runtime = better), we invert scores first so
    that faster programs get a larger slice of the wheel.

    Risk: one very fast individual can dominate the wheel (low diversity).
    """
    # Invert so lower runtime = higher selection weight
    max_score = max((s for s in fitness_scores if s != float('inf')), default=0)
    inverted = [max_score - s + 1e-6 if s != float('inf') else 0.0 for s in fitness_scores]
    total = sum(inverted)
    if total == 0:
        # All scores are inf — fall back to random selection
        return random.choices(population, k=n)
    probabilities = [w / total for w in inverted]
    return random.choices(population, weights=probabilities, k=n)


def selection_stochastic(population, fitness_scores, n):
    """
    Stochastic Universal Sampling (SUS).
    Like roulette but uses n equally-spaced pointers around the wheel
    in a single spin. Guarantees fairer coverage — no individual can
    be selected more than once per spin (unlike roulette).
    """
    if all(f == float('inf') for f in fitness_scores):
        return random.choices(population, k=n)
    max_score = max(fitness_scores)
    inverted = [max_score - s + 1e-6 for s in fitness_scores]
    total = sum(inverted)

    # Build cumulative probability wheel
    cumulative = []
    running = 0.0
    for w in inverted:
        running += w / total
        cumulative.append(running)

    # n equally spaced pointers starting from a single random offset
    step = 1.0 / n
    start = random.uniform(0, step)
    pointers = [start + i * step for i in range(n)]

    selected = []
    for pointer in pointers:
        for idx, cum_prob in enumerate(cumulative):
            if pointer <= cum_prob:
                selected.append(population[idx][:])
                break

    return selected


def selection_sigma_scaling(population, fitness_scores, n):
    """
    Sigma Scaling Selection.
    Adjusts selection pressure based on the population's standard deviation.
    Prevents premature convergence early on (when std is high) and
    maintains pressure later (when std is low and individuals are similar).

    Expected fitness of individual i = 1 + (f_avg - f_i) / (2 * sigma)
    (inverted because we minimise)
    """
    if all(f == float('inf') for f in fitness_scores):
        return random.choices(population, k=n)
    valid = [f for f in fitness_scores if f != float('inf')]

    avg   = statistics.mean(valid)
    try:
        sigma = statistics.stdev(valid)
    except statistics.StatisticsError:
        sigma = 1e-6

    if sigma < 1e-6:
        # All fitnesses are equal — fall back to uniform selection
        return random.choices(population, k=n)

    # Scale weights: individuals better than average get higher weight
    scaled = [max(1.0 + (avg - s) / (2.0 * sigma), 0.0) for s in fitness_scores]
    total = sum(scaled)
    if total == 0:
        return random.choices(population, k=n)

    probabilities = [w / total for w in scaled]
    return random.choices(population, weights=probabilities, k=n)


def selection_ranking(population, fitness_scores, n):
    """
    Rank-Based Selection.
    Sorts individuals by fitness and assigns selection probability based
    purely on rank (not raw fitness value). This prevents any single
    dominant individual from taking over the population.

    Rank 1 = best (lowest runtime). Each rank gets equal probability
    increment above the one below it.
    """
    # Pair each individual with its fitness, sort ascending (best first)
    ranked = sorted(zip(fitness_scores, population), key=lambda x: x[0])
    total_ranks = len(ranked)

    # Assign weights: rank 1 (best) gets weight = total_ranks, worst gets 1
    weights = list(range(total_ranks, 0, -1))
    sorted_population = [ind for _, ind in ranked]
    return random.choices(sorted_population, weights=weights, k=n)


def selection_linear_ranking(population, fitness_scores, n):
    """
    Linear Ranking Selection (used by FOGA — their best-performing selection).
    Like rank-based but the selection probability is a linear function of rank:
        P(i) = (2 - s) / N  +  2*rank_i*(s - 1) / (N*(N-1))
    where s is the selection pressure (typically 1.5–2.0).

    Key advantage over roulette: even the weakest individual has a non-zero
    chance of selection, maintaining genetic diversity throughout the run.
    This is why FOGA found it best — it helps discover flags that individually
    seem unimportant but work well in combination.
    """
    s = 1.8  # Selection pressure (1 < s <= 2); higher = more pressure on best
    N = len(population)
    # Sort ascending by fitness (index 0 = worst, index N-1 = best)
    ranked = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)

    weights = []
    for rank_i, _ in enumerate(ranked):
        # rank_i=0 is worst, rank_i=N-1 is best
        prob = (2 - s) / N + (2 * rank_i * (s - 1)) / (N * (N - 1))
        weights.append(max(prob, 0.0))

    sorted_population = [ind for _, ind in ranked]
    return random.choices(sorted_population, weights=weights, k=n)


def selection_tournament(population, fitness_scores, n):
    """
    Tournament Selection.
    Runs n independent mini-tournaments of size TOURNAMENT_SIZE.
    The individual with the lowest runtime in each tournament wins.
    Simple, fast, and tunable via TOURNAMENT_SIZE.
    """
    selected = []
    paired = list(zip(fitness_scores, population))
    for _ in range(n):
        competitors = random.choices(paired, k=TOURNAMENT_SIZE)
        winner = min(competitors, key=lambda x: x[0])
        selected.append(winner[1][:])
    return selected


def select_parents(population, fitness_scores):
    """
    Dispatcher: selects two parents using the chosen selection strategy.
    The number of candidates to draw is based on PARENTS_PORTION.
    """
    n_parents = max(2, int(len(population) * PARENTS_PORTION))

    if SELECTION_TYPE == 'fully_random':
        pool = selection_fully_random(population, fitness_scores, n_parents)
    elif SELECTION_TYPE == 'roulette':
        pool = selection_roulette(population, fitness_scores, n_parents)
    elif SELECTION_TYPE == 'stochastic':
        pool = selection_stochastic(population, fitness_scores, n_parents)
    elif SELECTION_TYPE == 'sigma_scaling':
        pool = selection_sigma_scaling(population, fitness_scores, n_parents)
    elif SELECTION_TYPE == 'ranking':
        pool = selection_ranking(population, fitness_scores, n_parents)
    elif SELECTION_TYPE == 'linear_ranking':
        pool = selection_linear_ranking(population, fitness_scores, n_parents)
    elif SELECTION_TYPE == 'tournament':
        pool = selection_tournament(population, fitness_scores, n_parents)
    else:
        raise ValueError(f"Unknown selection type: '{SELECTION_TYPE}'")

    # Pick two parents from the selected pool
    parent_a = random.choice(pool)
    parent_b = random.choice(pool)
    return parent_a, parent_b


# =============================================================================
# POPULATION INITIALISATION
# =============================================================================

def initialize_population(genome_length):
    """
    Creates the initial random population of chromosomes (lists of 0s/1s).
    """
    return [
        [random.randint(0, 1) for _ in range(genome_length)]
        for _ in range(POPULATION_SIZE)
    ]


# =============================================================================
# MAIN GA LOOP
# =============================================================================

def run_genetic_algorithm(benchmark_path, polybench=False, extra_sources=None, extra_includes=None, defines=None):
    """
    The main GA control loop.
    Automatically builds the flag list by scanning the benchmark source
    and applying CFSCA feature matching + PDCAT dependency constraints.
    """

    # Seed the RNG so all operators (crossover, mutation, selection, init) are reproducible
    random.seed(RANDOM_SEED)

    # Build the flag list for this specific benchmark
    core_flag_list = build_flag_list(CFSCA_FLAGS_FILE, benchmark_path)
    baseline = get_baseline_runtime(benchmark_path, polybench=polybench,
                                     extra_sources=extra_sources,
                                     extra_includes=extra_includes,
                                     defines=defines)
    print(f"\n--- Starting Genetic Algorithm Search ---")
    print(f"Baseline (-O3) : {baseline:.4f}s")
    print(f"Genome Length  : {len(core_flag_list)} flags")
    print(f"Benchmark      : {benchmark_path}")
    print(f"Genome Length  : {len(core_flag_list)} flags")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Crossover Type : {CROSSOVER_TYPE}  (rate={CROSSOVER_RATE})")
    print(f"Mutation Type  : {MUTATION_TYPE}  (rate={MUTATION_RATE})")
    print(f"Selection Type : {SELECTION_TYPE}")
    print(f"Elitism Ratio  : {ELITISM_RATIO}")
    print(f"Parents Portion: {PARENTS_PORTION}")
    print(f"Random Seed    : {RANDOM_SEED if RANDOM_SEED is not None else 'None (non-deterministic)'}\n")

    genome_length = len(core_flag_list)
    population = initialize_population(genome_length)

    best_chromosome = None
    best_fitness = float("inf")
    generations_without_improvement = 0
    log = []

    # Number of elite individuals to carry over unchanged each generation
    n_elite = max(1, int(POPULATION_SIZE * ELITISM_RATIO))

    for generation in range(MAX_GENERATIONS):
        start_time = time.time()

        # 1. EVALUATE FITNESS for entire population
        fitness_scores = [
            get_fitness_score(chromosome, core_flag_list, benchmark_path,
                  polybench=polybench,
                  extra_sources=extra_sources,
                  extra_includes=extra_includes,
                  defines=defines)
            for chromosome in population
        ]

        # 2. FIND BEST in current generation and update overall best
        ranked_pairs = sorted(zip(fitness_scores, population), key=lambda x: x[0])
        current_best_fitness = ranked_pairs[0][0]
        current_best_chromosome = ranked_pairs[0][1]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = current_best_chromosome[:]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        elapsed_time = time.time() - start_time
        valid_scores = [f for f in fitness_scores if f != float('inf')]
        log.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': statistics.mean(valid_scores) if valid_scores else float('inf'),
            'time': elapsed_time
        })

        print(
            f"Gen {generation+1:03d}/{MAX_GENERATIONS} | "
            f"Best: {best_fitness:.4f}s | "
            f"Avg: {log[-1]['avg_fitness']:.4f}s | "
            f"No Improve: {generations_without_improvement}/{MAX_NO_IMPROVEMENT} | "
            f"Elapsed: {elapsed_time:.2f}s"
        )

        # 3. EARLY STOPPING (MIWI — Max Iterations Without Improvement)
        if generations_without_improvement >= MAX_NO_IMPROVEMENT:
            print(f"\nEarly stop: no improvement for {MAX_NO_IMPROVEMENT} generations.")
            break

        # 4. BUILD NEW POPULATION
        new_population = []

        # --- Elitism: carry over the top n_elite individuals unchanged ---
        for _, elite_chrom in ranked_pairs[:n_elite]:
            new_population.append(elite_chrom[:])

        # --- Fill the rest via selection → crossover → mutation ---
        while len(new_population) < POPULATION_SIZE:
            parent_a, parent_b = select_parents(population, fitness_scores)
            offspring_a, offspring_b = crossover(parent_a, parent_b)

            new_population.append(mutate(offspring_a))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(offspring_b))

        population = new_population

# Print the optimal flag set found
    enabled_flags  = [core_flag_list[i] for i, g in enumerate(best_chromosome) if g == 1]
    disabled_flags = [core_flag_list[i] for i, g in enumerate(best_chromosome) if g == 0]

    print(f"\n{'='*60}")
    print(f"  OPTIMAL FLAG SET FOUND")
    print(f"  Best fitness : {best_fitness:.4f}s")
    print(f"{'='*60}")
    print(f"\n  Enabled ({len(enabled_flags)} flags):")
    for flag in enabled_flags:
        print(f"    -f{flag}")
    print(f"\n  Disabled ({len(disabled_flags)} flags — explicitly turned off):")
    for flag in disabled_flags:
        print(f"    -fno-{flag}")
    print(f"\n  Full GCC command:")
    flag_str = " ".join(
        [f"-f{f}" for f in enabled_flags] +
        [f"-fno-{f}" for f in disabled_flags]
    )
    print(f"    gcc -O3 {flag_str}")
    print(f"{'='*60}\n")

    return {
        "best_chromosome": best_chromosome,
        "best_fitness": best_fitness,
        "best_flags": enabled_flags,
        "log": log
    }

# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python genetic_algorithm.py <benchmark_path>")
#         sys.exit(1)

#     result = run_genetic_algorithm(sys.argv[1])