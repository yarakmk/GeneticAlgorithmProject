"""
Unit tests for sensitivity_analysis.py.

All subprocess and filesystem calls are mocked — no compiler or benchmark needed.

Note: sensitivity_analysis.py runs SDK_PATH = get_sdk_path() at import time.
This is fine on macOS (xcrun is available), but the actual value is patched
in tests that call measure_runtime so it doesn't affect results.
"""
import pytest
from unittest.mock import patch, MagicMock, call
import algorithm.sensitivity_analysis as sa


# ---------------------------------------------------------------------------
# get_sdk_path
# ---------------------------------------------------------------------------

class TestGetSdkPath:
    @patch('algorithm.sensitivity_analysis.subprocess.run')
    def test_returns_sdk_path_on_success(self, mock_run):
        mock_run.return_value = MagicMock(stdout='/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk\n')
        result = sa.get_sdk_path()
        assert result == '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'

    @patch('algorithm.sensitivity_analysis.subprocess.run', side_effect=Exception("xcrun not found"))
    def test_returns_none_on_failure(self, mock_run):
        result = sa.get_sdk_path()
        assert result is None


# ---------------------------------------------------------------------------
# measure_runtime
# ---------------------------------------------------------------------------

class TestMeasureRuntime:
    def _mock_successful_run(self, mock_run, mock_perf):
        """Configure mocks for a successful compile + warmup + N timed runs."""
        mock_run.return_value = MagicMock(returncode=0)
        # perf_counter called in pairs (start, end) for each timed run
        times = []
        for i in range(sa.TRIALS):
            times += [float(i), float(i) + 0.1]  # each run takes 0.1s
        mock_perf.side_effect = times

    @patch('algorithm.sensitivity_analysis.time.sleep')
    @patch('algorithm.sensitivity_analysis.time.perf_counter')
    @patch('algorithm.sensitivity_analysis.subprocess.run')
    @patch('algorithm.sensitivity_analysis.SDK_PATH', '/fake/sdk')
    def test_returns_float_on_success(self, mock_run, mock_perf, mock_sleep):
        self._mock_successful_run(mock_run, mock_perf)
        result = sa.measure_runtime(['-finline'])
        assert isinstance(result, float)
        assert result > 0

    @patch('algorithm.sensitivity_analysis.time.sleep')
    @patch('algorithm.sensitivity_analysis.time.perf_counter')
    @patch('algorithm.sensitivity_analysis.subprocess.run')
    @patch('algorithm.sensitivity_analysis.SDK_PATH', None)
    def test_works_without_sdk_path(self, mock_run, mock_perf, mock_sleep):
        self._mock_successful_run(mock_run, mock_perf)
        result = sa.measure_runtime(['-finline'])
        assert result is not None

    @patch('algorithm.sensitivity_analysis.subprocess.run', side_effect=Exception("compile failed"))
    @patch('algorithm.sensitivity_analysis.SDK_PATH', '/fake/sdk')
    def test_returns_none_on_compile_failure(self, mock_run):
        result = sa.measure_runtime(['-finline'])
        assert result is None

    @patch('algorithm.sensitivity_analysis.time.sleep')
    @patch('algorithm.sensitivity_analysis.time.perf_counter')
    @patch('algorithm.sensitivity_analysis.subprocess.run')
    @patch('algorithm.sensitivity_analysis.SDK_PATH', '/fake/sdk')
    def test_trims_outliers_with_enough_runs(self, mock_run, mock_perf, mock_sleep):
        """With TRIALS >= 5, fastest and slowest are trimmed."""
        mock_run.return_value = MagicMock(returncode=0)
        # Simulate run times: 0.1 (fast outlier), 0.2, 0.2, 0.2, 1.0 (slow outlier)
        run_times = [0.1, 0.2, 0.2, 0.2, 1.0]
        perf_side = []
        for t in run_times:
            perf_side += [0.0, t]
        mock_perf.side_effect = perf_side

        with patch.object(sa, 'TRIALS', 5):
            result = sa.measure_runtime(['-finline'])

        # After trimming [0.1, 1.0], mean of [0.2, 0.2, 0.2] = 0.2
        assert result == pytest.approx(0.2, abs=1e-9)

    @patch('algorithm.sensitivity_analysis.time.sleep')
    @patch('algorithm.sensitivity_analysis.time.perf_counter')
    @patch('algorithm.sensitivity_analysis.subprocess.run')
    @patch('algorithm.sensitivity_analysis.SDK_PATH', '/fake/sdk')
    def test_no_trim_with_few_runs(self, mock_run, mock_perf, mock_sleep):
        """With fewer than 5 runs, all times are used (no trimming)."""
        mock_run.return_value = MagicMock(returncode=0)
        run_times = [0.1, 0.3]
        perf_side = []
        for t in run_times:
            perf_side += [0.0, t]
        mock_perf.side_effect = perf_side

        with patch.object(sa, 'TRIALS', 2):
            result = sa.measure_runtime(['-finline'])

        assert result == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# run_sensitivity_test
# ---------------------------------------------------------------------------

class TestRunSensitivityTest:
    def _make_flags_file(self, tmp_path, flags):
        f = tmp_path / "flags.txt"
        f.write_text("\n".join(flags) + "\n")
        return str(f)

    def test_exits_if_flags_file_missing(self, capsys):
        with patch.object(sa, 'FLAGS_FILE', '/nonexistent/path.txt'):
            sa.run_sensitivity_test()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_exits_if_baseline_fails(self, tmp_path, capsys):
        flags_file = self._make_flags_file(tmp_path, ['inline'])
        with patch.object(sa, 'FLAGS_FILE', flags_file), \
             patch('algorithm.sensitivity_analysis.measure_runtime', return_value=None):
            sa.run_sensitivity_test()
        captured = capsys.readouterr()
        assert "baseline failed" in captured.out.lower()

    def test_writes_influential_flags_file(self, tmp_path, capsys):
        flags_file = self._make_flags_file(tmp_path, ['inline', 'unroll'])

        # inline gives 5% speedup, unroll gives 0% (no speedup)
        def fake_measure(flags):
            if flags == []:
                return 1.0
            if '-finline' in flags:
                return 0.95   # 5% faster → kept
            return 1.0        # 0% → dropped

        import os
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(sa, 'FLAGS_FILE', flags_file), \
                 patch.object(sa, 'RE_BASELINE_EVERY', 999), \
                 patch.object(sa, 'COOLDOWN_SECONDS', 0), \
                 patch('algorithm.sensitivity_analysis.measure_runtime', side_effect=fake_measure), \
                 patch('algorithm.sensitivity_analysis.time.sleep'):
                sa.run_sensitivity_test()
        finally:
            os.chdir(orig_dir)

        # influential_flags.txt should exist and contain 'inline' but not 'unroll'
        out = tmp_path / "influential_flags.txt"
        assert out.exists()
        content = out.read_text()
        assert 'inline' in content
        assert 'unroll' not in content

    def test_skips_failed_flag(self, tmp_path, capsys):
        """If measure_runtime returns None for a flag, it should be skipped gracefully."""
        flags_file = self._make_flags_file(tmp_path, ['good-flag', 'bad-flag'])

        def fake_measure(flags):
            if flags == []:
                return 1.0
            if '-fbad-flag' in flags:
                return None
            return 0.9

        import os
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(sa, 'FLAGS_FILE', flags_file), \
                 patch.object(sa, 'TRIALS', 1), \
                 patch.object(sa, 'RE_BASELINE_EVERY', 999), \
                 patch.object(sa, 'COOLDOWN_SECONDS', 0), \
                 patch('algorithm.sensitivity_analysis.measure_runtime', side_effect=fake_measure), \
                 patch('algorithm.sensitivity_analysis.time.sleep'):
                sa.run_sensitivity_test()
        finally:
            os.chdir(orig_dir)

        captured = capsys.readouterr()
        assert "FAILED" in captured.out

    def test_baseline_recalibration_triggered(self, tmp_path, capsys):
        """Lines 76-80: baseline is re-measured every RE_BASELINE_EVERY flags."""
        # 3 flags, RE_BASELINE_EVERY=2 → recalibration fires at i=2
        flags_file = self._make_flags_file(tmp_path, ['a', 'b', 'c'])
        calls = []

        def fake_measure(flags):
            calls.append(flags)
            return 1.0  # stable baseline, no drift

        import os
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch.object(sa, 'FLAGS_FILE', flags_file), \
                 patch.object(sa, 'RE_BASELINE_EVERY', 2), \
                 patch.object(sa, 'COOLDOWN_SECONDS', 0), \
                 patch('algorithm.sensitivity_analysis.measure_runtime', side_effect=fake_measure), \
                 patch('algorithm.sensitivity_analysis.time.sleep'):
                sa.run_sensitivity_test()
        finally:
            os.chdir(orig_dir)

        # One initial baseline call [] + 3 flag calls + 1 recalibration call []
        baseline_calls = [c for c in calls if c == []]
        assert len(baseline_calls) == 2  # initial + one recalibration


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

class TestMainBlock:
    def test_main_calls_run_sensitivity_test(self):
        import runpy
        # run_path re-executes the file in a fresh namespace. Patching
        # algorithm.sensitivity_analysis.run_sensitivity_test won't intercept
        # the call in that new namespace, so we patch FLAGS_FILE to point to a
        # non-existent file — run_sensitivity_test will just print an error and
        # return immediately, confirming the __main__ block executed without crash.
        with patch.object(sa, 'FLAGS_FILE', '/nonexistent/flags.txt'):
            runpy.run_path('algorithm/sensitivity_analysis.py', run_name='__main__')
