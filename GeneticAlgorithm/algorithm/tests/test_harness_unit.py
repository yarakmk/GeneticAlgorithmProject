"""
Unit tests for test_harness.py.

compile_program / run_program_and_time / get_fitness_score all shell out to
g++-15 and run real binaries, so we mock subprocess and os.path to keep
tests fast and portable.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import algorithm.test_harness as th


# ---------------------------------------------------------------------------
# translate_chromosome_to_flags
# ---------------------------------------------------------------------------

class TestTranslateChromosomeToFlags:
    def test_starts_with_O3(self):
        flags = th.translate_chromosome_to_flags([1, 0], ['inline', 'unroll'])
        assert flags[0] == '-O3'

    def test_gene_1_produces_f_flag(self):
        flags = th.translate_chromosome_to_flags([1, 0, 1], ['inline', 'unroll', 'vectorize'])
        assert '-finline' in flags
        assert '-fvectorize' in flags

    def test_gene_0_produces_fno_flag(self):
        flags = th.translate_chromosome_to_flags([1, 0, 1], ['inline', 'unroll', 'vectorize'])
        assert '-fno-unroll' in flags

    def test_output_length(self):
        chrom = [1, 0, 1, 0]
        flag_list = ['a', 'b', 'c', 'd']
        flags = th.translate_chromosome_to_flags(chrom, flag_list)
        # -O3 + one flag per gene
        assert len(flags) == 1 + len(chrom)

    def test_all_enabled(self):
        chrom = [1, 1, 1]
        flag_list = ['x', 'y', 'z']
        flags = th.translate_chromosome_to_flags(chrom, flag_list)
        assert flags == ['-O3', '-fx', '-fy', '-fz']

    def test_all_disabled(self):
        chrom = [0, 0, 0]
        flag_list = ['x', 'y', 'z']
        flags = th.translate_chromosome_to_flags(chrom, flag_list)
        assert flags == ['-O3', '-fno-x', '-fno-y', '-fno-z']

    def test_empty_chromosome(self):
        flags = th.translate_chromosome_to_flags([], [])
        assert flags == ['-O3']


# ---------------------------------------------------------------------------
# compile_program
# ---------------------------------------------------------------------------

class TestCompileProgram:
    @patch('algorithm.test_harness.subprocess.check_output', return_value=b'/sdk/path\n')
    @patch('algorithm.test_harness.subprocess.run')
    def test_returns_true_on_success(self, mock_run, mock_sdk):
        mock_run.return_value = MagicMock(returncode=0)
        result = th.compile_program('bench.cpp', ['-O3'], 'bench.out')
        assert result is True

    @patch('algorithm.test_harness.subprocess.check_output', return_value=b'/sdk/path\n')
    @patch('algorithm.test_harness.subprocess.run', side_effect=__import__('subprocess').CalledProcessError(1, 'g++', stderr='error'))
    def test_returns_false_on_failure(self, mock_run, mock_sdk):
        result = th.compile_program('bench.cpp', ['-O3'], 'bench.out')
        assert result is False

    @patch('algorithm.test_harness.subprocess.check_output', side_effect=__import__('subprocess').CalledProcessError(1, 'xcrun'))
    @patch('algorithm.test_harness.subprocess.run')
    def test_handles_missing_sdk(self, mock_run, mock_sdk):
        """Should still attempt compilation even if xcrun fails."""
        mock_run.return_value = MagicMock(returncode=0)
        result = th.compile_program('bench.cpp', ['-O3'], 'bench.out')
        assert result is True


# ---------------------------------------------------------------------------
# run_program_and_time
# ---------------------------------------------------------------------------

class TestRunProgramAndTime:
    @patch('algorithm.test_harness.os.path.exists', return_value=False)
    def test_returns_inf_if_executable_missing(self, mock_exists):
        result = th.run_program_and_time('./missing.out')
        assert result == float('inf')

    @patch('algorithm.test_harness.os.path.exists', return_value=True)
    @patch('algorithm.test_harness.subprocess.run')
    @patch('algorithm.test_harness.time.perf_counter', side_effect=[0.0, 0.5])
    def test_returns_elapsed_time(self, mock_time, mock_run, mock_exists):
        mock_run.return_value = MagicMock(returncode=0)
        result = th.run_program_and_time('./bench.out', polybench=False)
        assert result == pytest.approx(0.5)

    @patch('algorithm.test_harness.os.path.exists', return_value=True)
    @patch('algorithm.test_harness.subprocess.run')
    def test_polybench_reads_stdout(self, mock_run, mock_exists):
        mock_run.return_value = MagicMock(returncode=0, stdout='0.1234\n')
        result = th.run_program_and_time('./bench.out', polybench=True)
        assert result == pytest.approx(0.1234)

    @patch('algorithm.test_harness.os.path.exists', return_value=True)
    @patch('algorithm.test_harness.subprocess.run', side_effect=__import__('subprocess').CalledProcessError(1, './bench.out'))
    def test_returns_inf_on_crash(self, mock_run, mock_exists):
        result = th.run_program_and_time('./bench.out')
        assert result == float('inf')

    @patch('algorithm.test_harness.os.path.exists', return_value=True)
    @patch('algorithm.test_harness.subprocess.run', side_effect=__import__('subprocess').TimeoutExpired('./bench.out', 30))
    def test_returns_inf_on_timeout(self, mock_run, mock_exists):
        result = th.run_program_and_time('./bench.out')
        assert result == float('inf')

    @patch('algorithm.test_harness.os.path.exists', return_value=True)
    @patch('algorithm.test_harness.subprocess.run')
    def test_polybench_invalid_stdout_returns_inf(self, mock_run, mock_exists):
        mock_run.return_value = MagicMock(returncode=0, stdout='not_a_number\n')
        result = th.run_program_and_time('./bench.out', polybench=True)
        assert result == float('inf')


# ---------------------------------------------------------------------------
# get_fitness_score (integration of compile + run, fully mocked)
# ---------------------------------------------------------------------------

class TestGetFitnessScore:

    def test_returns_median_of_runs(self):
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        call_count = {'n': 0}

        def fake_run(exe, polybench=False):
            t = times[call_count['n']]
            call_count['n'] += 1
            return t

        with patch('algorithm.test_harness.compile_program', return_value=True), \
             patch('algorithm.test_harness.run_program_and_time', side_effect=fake_run), \
             patch('algorithm.test_harness.os.path.exists', return_value=True), \
             patch('algorithm.test_harness.os.remove'), \
             patch.object(th, 'N_RUNS', 5):
            result = th.get_fitness_score([1, 0], ['inline', 'unroll'], 'bench/bench.cpp')

        assert result == pytest.approx(0.3)  # median of [0.1,0.2,0.3,0.4,0.5]

    def test_returns_inf_on_compile_failure(self):
        with patch('algorithm.test_harness.compile_program', return_value=False):
            result = th.get_fitness_score([1, 0], ['inline', 'unroll'], 'bench/bench.cpp')
        assert result == float('inf')

    def test_returns_inf_if_any_run_fails(self):
        run_results = [0.2, float('inf')]
        call_count = {'n': 0}

        def fake_run(exe, polybench=False):
            t = run_results[call_count['n'] % len(run_results)]
            call_count['n'] += 1
            return t

        with patch('algorithm.test_harness.compile_program', return_value=True), \
             patch('algorithm.test_harness.run_program_and_time', side_effect=fake_run), \
             patch('algorithm.test_harness.os.path.exists', return_value=True), \
             patch('algorithm.test_harness.os.remove'), \
             patch.object(th, 'N_RUNS', 2):
            result = th.get_fitness_score([1, 0], ['inline', 'unroll'], 'bench/bench.cpp')

        assert result == float('inf')


# ---------------------------------------------------------------------------
# get_baseline_runtime
# ---------------------------------------------------------------------------

class TestGetBaselineRuntime:
    def test_returns_median_of_runs(self, capsys):
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        call_count = {'n': 0}

        def fake_run(exe, polybench=False):
            t = times[call_count['n']]
            call_count['n'] += 1
            return t

        with patch('algorithm.test_harness.compile_program', return_value=True), \
             patch('algorithm.test_harness.run_program_and_time', side_effect=fake_run), \
             patch('algorithm.test_harness.os.path.exists', return_value=True), \
             patch('algorithm.test_harness.os.remove'), \
             patch.object(th, 'N_RUNS', 5):
            result = th.get_baseline_runtime('bench/bench.cpp')

        assert result == pytest.approx(0.3)

    def test_returns_inf_on_compile_failure(self, capsys):
        with patch('algorithm.test_harness.compile_program', return_value=False):
            result = th.get_baseline_runtime('bench/bench.cpp')
        assert result == float('inf')

    def test_returns_inf_if_any_run_fails(self, capsys):
        run_results = [0.2, float('inf')]
        call_count = {'n': 0}

        def fake_run(exe, polybench=False):
            t = run_results[call_count['n'] % len(run_results)]
            call_count['n'] += 1
            return t

        with patch('algorithm.test_harness.compile_program', return_value=True), \
             patch('algorithm.test_harness.run_program_and_time', side_effect=fake_run), \
             patch('algorithm.test_harness.os.path.exists', return_value=True), \
             patch('algorithm.test_harness.os.remove'), \
             patch.object(th, 'N_RUNS', 2):
            result = th.get_baseline_runtime('bench/bench.cpp')

        assert result == float('inf')

    def test_uses_only_O3_flag(self, capsys):
        """Baseline must compile with only -O3, no extra flags."""
        compile_calls = []

        def fake_compile(src, flags, exe, *args, **kwargs):
            compile_calls.append(flags[:])
            return True

        with patch('algorithm.test_harness.compile_program', side_effect=fake_compile), \
             patch('algorithm.test_harness.run_program_and_time', return_value=0.1), \
             patch('algorithm.test_harness.os.path.exists', return_value=True), \
             patch('algorithm.test_harness.os.remove'), \
             patch.object(th, 'N_RUNS', 1):
            th.get_baseline_runtime('bench/bench.cpp')

        assert compile_calls[0] == ['-O3']
