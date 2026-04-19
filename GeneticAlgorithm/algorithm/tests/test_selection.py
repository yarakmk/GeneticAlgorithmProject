import pytest
import algorithm.geneticAlgorithm as ga

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def population():
    """Small population of 6 chromosomes, length 4."""
    return [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ]


@pytest.fixture
def fitness_scores():
    """Distinct fitness scores (lower = better runtime)."""
    return [0.5, 1.2, 0.8, 2.0, 0.3, 1.5]


@pytest.fixture
def all_inf_scores(population):
    return [float('inf')] * len(population)


def from_population(selected, population):
    """Check that every selected chromosome exists in the original population."""
    pop_set = [tuple(c) for c in population]
    return all(tuple(s) in pop_set for s in selected)


# ---------------------------------------------------------------------------
# Fully Random
# ---------------------------------------------------------------------------

class TestSelectionFullyRandom:
    def test_returns_n_individuals(self, population, fitness_scores):
        result = ga.selection_fully_random(population, fitness_scores, 4)
        assert len(result) == 4

    def test_individuals_from_population(self, population, fitness_scores):
        result = ga.selection_fully_random(population, fitness_scores, 4)
        assert from_population(result, population)


# ---------------------------------------------------------------------------
# Roulette
# ---------------------------------------------------------------------------

class TestSelectionRoulette:
    def test_returns_n_individuals(self, population, fitness_scores):
        result = ga.selection_roulette(population, fitness_scores, 4)
        assert len(result) == 4

    def test_individuals_from_population(self, population, fitness_scores):
        result = ga.selection_roulette(population, fitness_scores, 4)
        assert from_population(result, population)

    def test_all_inf_falls_back(self, population, all_inf_scores):
        result = ga.selection_roulette(population, all_inf_scores, 3)
        assert len(result) == 3
        assert from_population(result, population)


# ---------------------------------------------------------------------------
# Stochastic Universal Sampling
# ---------------------------------------------------------------------------

class TestSelectionStochastic:
    def test_returns_n_individuals(self, population, fitness_scores):
        result = ga.selection_stochastic(population, fitness_scores, 4)
        assert len(result) == 4

    def test_individuals_from_population(self, population, fitness_scores):
        result = ga.selection_stochastic(population, fitness_scores, 4)
        assert from_population(result, population)

    def test_all_inf_falls_back(self, population, all_inf_scores):
        result = ga.selection_stochastic(population, all_inf_scores, 3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Sigma Scaling
# ---------------------------------------------------------------------------

class TestSelectionSigmaScaling:
    def test_returns_n_individuals(self, population, fitness_scores):
        result = ga.selection_sigma_scaling(population, fitness_scores, 4)
        assert len(result) == 4

    def test_individuals_from_population(self, population, fitness_scores):
        result = ga.selection_sigma_scaling(population, fitness_scores, 4)
        assert from_population(result, population)

    def test_all_inf_falls_back(self, population, all_inf_scores):
        result = ga.selection_sigma_scaling(population, all_inf_scores, 3)
        assert len(result) == 3

    def test_identical_scores_falls_back(self, population):
        """When all fitnesses are equal, sigma=0 — should fall back gracefully."""
        equal_scores = [1.0] * len(population)
        result = ga.selection_sigma_scaling(population, equal_scores, 3)
        assert len(result) == 3
        assert from_population(result, population)

    def test_single_valid_score_triggers_statistics_error(self, population):
        """With only one valid score, stdev raises StatisticsError — sigma falls back to 1e-6."""
        # Five inf + one real score → stdev of a 1-element list raises StatisticsError
        scores = [float('inf')] * 5 + [0.5]
        result = ga.selection_sigma_scaling(population, scores, 3)
        assert len(result) == 3
        assert from_population(result, population)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestSelectionRanking:
    def test_returns_n_individuals(self, population, fitness_scores):
        result = ga.selection_ranking(population, fitness_scores, 4)
        assert len(result) == 4

    def test_individuals_from_population(self, population, fitness_scores):
        result = ga.selection_ranking(population, fitness_scores, 4)
        assert from_population(result, population)

    def test_best_individual_most_likely_selected(self, population):
        """With extreme fitness differences the best individual should dominate."""
        scores = [0.001, 100.0, 100.0, 100.0, 100.0, 100.0]
        counts = {tuple(c): 0 for c in population}
        for _ in range(1000):
            selected = ga.selection_ranking(population, scores, 1)
            counts[tuple(selected[0])] += 1
        best = tuple(population[0])   # weight 6/21 ≈ 29%, expected ~286
        worst = tuple(population[-1]) # weight 1/21 ≈  5%, expected ~48
        assert counts[best] > counts[worst]


# ---------------------------------------------------------------------------
# Linear Ranking
# ---------------------------------------------------------------------------

class TestSelectionLinearRanking:
    def test_returns_n_individuals(self, population, fitness_scores):
        result = ga.selection_linear_ranking(population, fitness_scores, 4)
        assert len(result) == 4

    def test_individuals_from_population(self, population, fitness_scores):
        result = ga.selection_linear_ranking(population, fitness_scores, 4)
        assert from_population(result, population)

    def test_all_individuals_have_nonzero_chance(self, population, fitness_scores):
        """Every individual should appear at least once in 500 draws."""
        seen = set()
        for _ in range(500):
            selected = ga.selection_linear_ranking(population, fitness_scores, 1)
            seen.add(tuple(selected[0]))
        assert len(seen) == len(population)


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

class TestSelectionTournament:
    def test_returns_n_individuals(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'TOURNAMENT_SIZE', 2)
        result = ga.selection_tournament(population, fitness_scores, 4)
        assert len(result) == 4

    def test_individuals_from_population(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'TOURNAMENT_SIZE', 2)
        result = ga.selection_tournament(population, fitness_scores, 4)
        assert from_population(result, population)

    def test_tournament_size_1_is_random(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'TOURNAMENT_SIZE', 1)
        result = ga.selection_tournament(population, fitness_scores, 6)
        assert len(result) == 6

    def test_winner_has_lower_fitness(self, population, monkeypatch):
        """The individual with the lowest fitness should win most tournaments."""
        monkeypatch.setattr(ga, 'TOURNAMENT_SIZE', 3)
        scores = [0.5, 1.2, 0.8, 2.0, 0.3, 1.5]
        best = tuple(population[4])  # score 0.3 is the best
        worst = tuple(population[3])  # score 2.0 is the worst
        counts = {best: 0, worst: 0}
        for _ in range(300):
            result = ga.selection_tournament(population, scores, 1)
            key = tuple(result[0])
            if key in counts:
                counts[key] += 1
        # Best individual should be selected far more often than the worst
        assert counts[best] > counts[worst]


# ---------------------------------------------------------------------------
# select_parents dispatcher
# ---------------------------------------------------------------------------

class TestSelectParents:
    def test_returns_two_parents(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'linear_ranking')
        monkeypatch.setattr(ga, 'PARENTS_PORTION', 0.5)
        pa, pb = ga.select_parents(population, fitness_scores)
        assert len(pa) == len(population[0])
        assert len(pb) == len(population[0])

    def test_parents_from_population(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'ranking')
        monkeypatch.setattr(ga, 'PARENTS_PORTION', 0.5)
        pa, pb = ga.select_parents(population, fitness_scores)
        assert from_population([pa, pb], population)

    def test_dispatches_fully_random(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'fully_random')
        monkeypatch.setattr(ga, 'PARENTS_PORTION', 0.5)
        pa, pb = ga.select_parents(population, fitness_scores)
        assert from_population([pa, pb], population)

    def test_dispatches_roulette(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'roulette')
        monkeypatch.setattr(ga, 'PARENTS_PORTION', 0.5)
        pa, pb = ga.select_parents(population, fitness_scores)
        assert from_population([pa, pb], population)

    def test_dispatches_stochastic(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'stochastic')
        monkeypatch.setattr(ga, 'PARENTS_PORTION', 0.5)
        pa, pb = ga.select_parents(population, fitness_scores)
        assert from_population([pa, pb], population)

    def test_dispatches_sigma_scaling(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'sigma_scaling')
        monkeypatch.setattr(ga, 'PARENTS_PORTION', 0.5)
        pa, pb = ga.select_parents(population, fitness_scores)
        assert from_population([pa, pb], population)

    def test_dispatches_tournament(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'tournament')
        monkeypatch.setattr(ga, 'PARENTS_PORTION', 0.5)
        monkeypatch.setattr(ga, 'TOURNAMENT_SIZE', 2)
        pa, pb = ga.select_parents(population, fitness_scores)
        assert from_population([pa, pb], population)

    def test_unknown_type_raises(self, population, fitness_scores, monkeypatch):
        monkeypatch.setattr(ga, 'SELECTION_TYPE', 'nonexistent')
        with pytest.raises(ValueError):
            ga.select_parents(population, fitness_scores)


# ---------------------------------------------------------------------------
# initialize_population
# ---------------------------------------------------------------------------

class TestInitializePopulation:
    def test_correct_population_size(self, monkeypatch):
        monkeypatch.setattr(ga, 'POPULATION_SIZE', 10)
        pop = ga.initialize_population(6)
        assert len(pop) == 10

    def test_correct_genome_length(self, monkeypatch):
        monkeypatch.setattr(ga, 'POPULATION_SIZE', 5)
        pop = ga.initialize_population(8)
        assert all(len(c) == 8 for c in pop)

    def test_all_genes_binary(self, monkeypatch):
        monkeypatch.setattr(ga, 'POPULATION_SIZE', 20)
        pop = ga.initialize_population(10)
        for chrom in pop:
            assert all(g in {0, 1} for g in chrom)

    def test_population_has_diversity(self, monkeypatch):
        """With 50 individuals of length 20, not all chromosomes should be identical."""
        monkeypatch.setattr(ga, 'POPULATION_SIZE', 50)
        pop = ga.initialize_population(20)
        unique = {tuple(c) for c in pop}
        assert len(unique) > 1
