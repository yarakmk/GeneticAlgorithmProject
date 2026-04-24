"""
Unit tests for main.py.

All external calls (run_genetic_algorithm, get_baseline_runtime) are mocked —
no compiler or benchmark files needed.
"""
import runpy
import pytest
from unittest.mock import patch

GOOD_RESULT = {
    'best_fitness':    0.5,
    'best_chromosome': [1, 0, 1],
    'best_flags':      ['inline', 'vectorize'],
    'log':             [],
}


# ---------------------------------------------------------------------------
# Baseline failure — benchmark is skipped
# ---------------------------------------------------------------------------

class TestBaselineFailure:

    def test_skips_benchmark_when_baseline_inf(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm') as mock_ga, \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=float('inf')):
            runpy.run_path('main.py', run_name='__main__')
        mock_ga.assert_not_called()

    def test_prints_error_when_baseline_inf(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm'), \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=float('inf')):
            runpy.run_path('main.py', run_name='__main__')
        captured = capsys.readouterr()
        assert 'ERROR' in captured.out


# ---------------------------------------------------------------------------
# GA result handling
# ---------------------------------------------------------------------------

class TestGAResults:

    def test_calls_run_genetic_algorithm_for_each_benchmark(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value=GOOD_RESULT) as mock_ga, \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=1.0):
            runpy.run_path('main.py', run_name='__main__')
        # main.py has 5 benchmarks — one call per benchmark
        assert mock_ga.call_count == 5

    def test_passes_polybench_true(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value=GOOD_RESULT) as mock_ga, \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=1.0):
            runpy.run_path('main.py', run_name='__main__')
        for c in mock_ga.call_args_list:
            assert c[1].get('polybench') is True

    def test_prints_speedup_when_ga_succeeds(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value=GOOD_RESULT), \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=1.0):
            runpy.run_path('main.py', run_name='__main__')
        captured = capsys.readouterr()
        assert 'Speedup' in captured.out

    def test_prints_no_converge_when_ga_returns_inf(self, capsys):
        inf_result = {**GOOD_RESULT, 'best_fitness': float('inf')}
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value=inf_result), \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=1.0):
            runpy.run_path('main.py', run_name='__main__')
        captured = capsys.readouterr()
        assert 'did not converge' in captured.out

    def test_ga_exception_does_not_crash_main(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   side_effect=RuntimeError("boom")), \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=1.0):
            runpy.run_path('main.py', run_name='__main__')
        captured = capsys.readouterr()
        assert 'FATAL ERROR' in captured.out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

class TestSummaryTable:

    def test_summary_printed_when_results_exist(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value=GOOD_RESULT), \
             patch('algorithm.test_harness.get_baseline_runtime', return_value=1.0):
            runpy.run_path('main.py', run_name='__main__')
        captured = capsys.readouterr()
        assert 'FINAL SUMMARY' in captured.out

    def test_no_summary_when_all_baselines_fail(self, capsys):
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value=GOOD_RESULT), \
             patch('algorithm.test_harness.get_baseline_runtime',
                   return_value=float('inf')):
            runpy.run_path('main.py', run_name='__main__')
        captured = capsys.readouterr()
        assert 'FINAL SUMMARY' not in captured.out
