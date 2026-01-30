import random
import statistics
import time
# Import the fitness function from the robust Test Harness script
from algorithm.test_harness import get_fitness_score 

# --- GA PARAMETERS (Tuning these is part of your research) ---
POPULATION_SIZE = 50          # Number of chromosomes in each generation
MAX_GENERATIONS = 10         # Maximum number of generations to run
CROSSOVER_RATE = 0.8          # Probability of crossover
MUTATION_RATE = 0.02          # Probability of a bit-flip mutation
TOURNAMENT_SIZE = 5           # Size of the pool for selection
MAX_NO_IMPROVEMENT = 20       # Stop if no better solution found after 20 generations
# --- END GA PARAMETERS ---


def initialize_population(genome_length):
    """
    Creates the initial random population of chromosomes (lists of 0s/1s).
    """
    population = []
    for _ in range(POPULATION_SIZE):
        # Create a new chromosome with random 0 or 1 for each gene
        chromosome = [random.randint(0, 1) for _ in range(genome_length)]
        population.append(chromosome)
    return population


def select_parents(population, fitness_scores):
    """
    Selects two parents using Tournament Selection (best fitness = lowest runtime).
    """
    def tournament_select():
        # Randomly select chromosomes for the tournament
        tournament_pool = random.choices(
            list(zip(population, fitness_scores)), 
            k=TOURNAMENT_SIZE
        )
        
        # Find the winner: the chromosome with the MINIMUM runtime (lowest is best)
        # tournament_pool element is (chromosome, score)
        winner = min(tournament_pool, key=lambda x: x[1])
        return winner[0] # Return the winning chromosome
        
    parent_a = tournament_select()
    parent_b = tournament_select()
    return parent_a, parent_b


def crossover(parent_a, parent_b):
    """
    Performs Single-Point Crossover with a given probability.
    """
    if random.random() < CROSSOVER_RATE:
        # Choose a random split point (crossover point)
        split_point = random.randint(1, len(parent_a) - 1)
        
        # Create two offspring by swapping the tails
        offspring_a = parent_a[:split_point] + parent_b[split_point:]
        offspring_b = parent_b[:split_point] + parent_a[split_point:]
        return offspring_a, offspring_b
    else:
        # No crossover: parents are copied as offspring
        return parent_a[:], parent_b[:]


def mutate(chromosome):
    """
    Performs Bit-Flip Mutation on the chromosome with a given probability.
    """
    mutated_chromosome = chromosome[:] # Start with a copy
    for i in range(len(mutated_chromosome)):
        if random.random() < MUTATION_RATE:
            # Flip the bit: 0 -> 1, 1 -> 0
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome


def run_genetic_algorithm(core_flag_list):
    """
    The main GA control loop.
    """
    print(f"\n--- Starting Genetic Algorithm Search ---")
    print(f"Genome Length: {len(core_flag_list)} flags (2^{len(core_flag_list)} search space)")
    print(f"Population Size: {POPULATION_SIZE}, Generations: {MAX_GENERATIONS}\n")

    genome_length = len(core_flag_list)
    population = initialize_population(genome_length)
    
    best_chromosome = None
    best_fitness = float("inf")
    generations_without_improvement = 0
    
    # Store results for plotting/analysis later
    log = [] 

    for generation in range(MAX_GENERATIONS):
        start_time = time.time()
        
        # 1. EVALUATE FITNESS
        # Run the Test Harness (the most time-consuming step) for the entire population
        fitness_scores = [
            get_fitness_score(chromosome, core_flag_list) 
            for chromosome in population
        ]
        
        # 2. FIND BEST IN CURRENT GENERATION (and update overall best)
        current_best_index = fitness_scores.index(min(fitness_scores))
        current_best_fitness = fitness_scores[current_best_index]
        current_best_chromosome = population[current_best_index]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = current_best_chromosome
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # Log the generation results
        elapsed_time = time.time() - start_time
        log.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': statistics.mean([f for f in fitness_scores if f != float('inf')]),
            'time': elapsed_time
        })
        
        print(f"Gen {generation+1:03d} | Best Time: {best_fitness:.4f}s | Avg Time: {log[-1]['avg_fitness']:.4f}s | No Improve: {generations_without_improvement}/{MAX_NO_IMPROVEMENT} | Time: {elapsed_time:.2f}s")


        # 3. CHECK STOP CONDITION
        if generations_without_improvement >= MAX_NO_IMPROVEMENT:
            print(f"\nStop condition met: No improvement for {MAX_NO_IMPROVEMENT} generations.")
            break


        # 4. GENERATE NEW POPULATION (Selection, Crossover, Mutation)
        new_population = []
        # Elitism: Automatically carry over the best chromosome to the next generation
        new_population.append(best_chromosome) 
        
        while len(new_population) < POPULATION_SIZE:
            parent_a, parent_b = select_parents(population, fitness_scores)
            
            offspring_a, offspring_b = crossover(parent_a, parent_b)
            
            # Mutate and add to the new population
            new_population.append(mutate(offspring_a))
            
            if len(new_population) < POPULATION_SIZE:
                 new_population.append(mutate(offspring_b))
        
        population = new_population

    return {
        "best_chromosome": best_chromosome,
        "best_fitness": best_fitness,
        "log": log
    }