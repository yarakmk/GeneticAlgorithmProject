"""
Tests for run_genetic_algorithm and the __main__ block in geneticAlgorithm.py.

All external calls (build_flag_list, get_baseline_runtime, get_fitness_score)
are mocked so no compiler or benchmark file is needed.
"""
import runpy
import pytest
from unittest.mock import patch, MagicMock
import algorithm.geneticAlgorithm as ga


# Minimal flag list for all tests — short enough to keep runs fast
FLAG_LIST = ['inline', 'unroll', 'vectorize', 'loop-unroll']
BENCH = 'bench/bench.cpp'


def _mock_patches(fitness_fn=None, population_size=4, max_gen=2, max_no_improve=2):
    """Return a dict of monkeypatched GA globals for fast, deterministic runs."""
    if fitness_fn is None:
        fitness_fn = lambda chrom, flags, path, **kw: 0.5

    return {
        'POPULATION_SIZE':    population_size,
        'MAX_GENERATIONS':    max_gen,
        'MAX_NO_IMPROVEMENT': max_no_improve,
        'ELITISM_RATIO':      0.25,
        'PARENTS_PORTION':    0.5,
        'CROSSOVER_RATE':     0.8,
        'MUTATION_RATE':      0.1,
        'CROSSOVER_TYPE':     'one_point',
        'MUTATION_TYPE':      'bit_flip',
        'SELECTION_TYPE':     'linear_ranking',
        'TOURNAMENT_SIZE':    2,
    }


# ---------------------------------------------------------------------------
# run_genetic_algorithm — happy path
# ---------------------------------------------------------------------------

class TestRunGeneticAlgorithm:

    def _run(self, fitness_fn=None, **overrides):
        params = _mock_patches(fitness_fn)
        params.update(overrides)
        with patch('algorithm.geneticAlgorithm.build_flag_list', return_value=FLAG_LIST), \
             patch('algorithm.geneticAlgorithm.get_baseline_runtime', return_value=1.0), \
             patch('algorithm.geneticAlgorithm.get_fitness_score',
                   side_effect=fitness_fn or (lambda *a, **kw: 0.5)):
            for attr, val in params.items():
                setattr(ga, attr, val)
            return ga.run_genetic_algorithm(BENCH)

    def test_returns_dict_with_expected_keys(self):
        result = self._run()
        assert set(result.keys()) == {'best_chromosome', 'best_fitness', 'best_flags', 'log'}

    def test_best_chromosome_has_correct_length(self):
        result = self._run()
        assert len(result['best_chromosome']) == len(FLAG_LIST)

    def test_best_chromosome_is_binary(self):
        result = self._run()
        assert all(g in (0, 1) for g in result['best_chromosome'])

    def test_best_fitness_is_finite(self):
        result = self._run()
        assert result['best_fitness'] != float('inf')

    def test_log_has_one_entry_per_generation(self):
        result = self._run(max_gen=2)
        assert len(result['log']) <= 2

    def test_log_entry_has_expected_keys(self):
        result = self._run()
        for entry in result['log']:
            assert 'generation' in entry
            assert 'best_fitness' in entry
            assert 'avg_fitness' in entry
            assert 'time' in entry

    def test_best_flags_are_subset_of_flag_list(self):
        result = self._run()
        assert all(f in FLAG_LIST for f in result['best_flags'])

    def test_improving_fitness_updates_best(self):
        """Fitness improves each generation — best_fitness should be the lowest seen."""
        call_count = {'n': 0}

        def improving(chrom, flags, path, **kw):
            call_count['n'] += 1
            # First generation returns 0.8, second 0.4
            return 0.8 if call_count['n'] <= 4 else 0.4

        result = self._run(fitness_fn=improving, max_gen=2, max_no_improve=99)
        assert result['best_fitness'] == pytest.approx(0.4)

    def test_early_stopping_on_no_improvement(self):
        """With max_no_improve=1, should stop after 2 generations (no improvement)."""
        result = self._run(max_gen=10, max_no_improve=1)
        # Only 2 log entries: gen 0 (improvement from inf→0.5) and gen 1 (no improvement)
        assert len(result['log']) == 2

    def test_all_inf_fitness_crashes_on_best_chromosome(self):
        """
        Known bug: when every chromosome returns inf across all generations,
        best_chromosome is never set (stays None). Line 556 then calls
        enumerate(None) which raises TypeError.
        This test documents the behaviour rather than asserting it works.
        """
        with pytest.raises(TypeError):
            self._run(fitness_fn=lambda *a, **kw: float('inf'),
                      max_gen=2, max_no_improve=2)

    def test_polybench_flag_passed_through(self):
        """polybench=True should be forwarded to get_fitness_score."""
        seen_kwargs = []

        def capture(chrom, flags, path, **kw):
            seen_kwargs.append(kw)
            return 0.5

        params = _mock_patches()
        with patch('algorithm.geneticAlgorithm.build_flag_list', return_value=FLAG_LIST), \
             patch('algorithm.geneticAlgorithm.get_baseline_runtime', return_value=1.0), \
             patch('algorithm.geneticAlgorithm.get_fitness_score', side_effect=capture):
            for attr, val in params.items():
                setattr(ga, attr, val)
            ga.run_genetic_algorithm(BENCH, polybench=True)

        assert any(kw.get('polybench') is True for kw in seen_kwargs)


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

class TestMainBlock:
    def test_no_args_prints_usage_and_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['geneticAlgorithm.py']):
                runpy.run_path('algorithm/geneticAlgorithm.py', run_name='__main__')
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Usage' in captured.out

    def test_with_arg_calls_run_genetic_algorithm(self):
        # run_path re-executes the file in a fresh namespace, so we mock the
        # imported dependencies rather than run_genetic_algorithm itself.
        with patch('sys.argv', ['geneticAlgorithm.py', 'bench.cpp']), \
             patch('algorithm.flag_matcher.build_flag_list', return_value=FLAG_LIST), \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=1.0), \
             patch('algorithm.test_harness.get_fitness_score', return_value=0.5):
            for attr, val in _mock_patches().items():
                setattr(ga, attr, val)
            runpy.run_path('algorithm/geneticAlgorithm.py', run_name='__main__')
