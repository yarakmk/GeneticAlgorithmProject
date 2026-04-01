"""
Unit tests for flag_matcher.py.

All file I/O uses pytest's tmp_path fixture — no real CFSCA_flags.txt needed.
"""
import pytest
import algorithm.flag_matcher as fm


# ---------------------------------------------------------------------------
# load_cfsca_flags
# ---------------------------------------------------------------------------

class TestLoadCFSCAFlags:
    def test_parses_categories_and_flags(self, tmp_path):
        f = tmp_path / "flags.txt"
        f.write_text("# loops\nloop-unroll\npeel-loops\n# branches\ntree-if-conversion\n")
        result = fm.load_cfsca_flags(str(f))
        assert result == {
            'loops': ['loop-unroll', 'peel-loops'],
            'branches': ['tree-if-conversion'],
        }

    def test_ignores_blank_lines(self, tmp_path):
        f = tmp_path / "flags.txt"
        f.write_text("# loops\n\nloop-unroll\n\npeel-loops\n")
        result = fm.load_cfsca_flags(str(f))
        assert result == {'loops': ['loop-unroll', 'peel-loops']}

    def test_empty_file_returns_empty_dict(self, tmp_path):
        f = tmp_path / "flags.txt"
        f.write_text("")
        result = fm.load_cfsca_flags(str(f))
        assert result == {}

    def test_flags_before_first_category_ignored(self, tmp_path):
        f = tmp_path / "flags.txt"
        f.write_text("orphan-flag\n# loops\nloop-unroll\n")
        result = fm.load_cfsca_flags(str(f))
        assert list(result.keys()) == ['loops']
        assert 'orphan-flag' not in result.get('loops', [])

    def test_multiple_flags_per_category(self, tmp_path):
        f = tmp_path / "flags.txt"
        f.write_text("# floats\nfloat-store\nassociative-math\nno-signed-zeros\n")
        result = fm.load_cfsca_flags(str(f))
        assert len(result['floats']) == 3

    def test_category_name_strips_hash_and_whitespace(self, tmp_path):
        f = tmp_path / "flags.txt"
        f.write_text("#   loops   \nloop-unroll\n")
        result = fm.load_cfsca_flags(str(f))
        assert 'loops' in result


# ---------------------------------------------------------------------------
# scan_benchmark
# ---------------------------------------------------------------------------

class TestScanBenchmark:
    def test_detects_for_loop(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("for (int i = 0; i < n; i++) { x += i; }")
        assert fm.scan_benchmark(str(f))['loops'] is True

    def test_detects_while_loop(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("while (x > 0) { x--; }")
        assert fm.scan_benchmark(str(f))['loops'] is True

    def test_detects_if_branch(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("if (x > 0) { return 1; }")
        assert fm.scan_benchmark(str(f))['branches'] is True

    def test_detects_double(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("double x = 3.14;")
        assert fm.scan_benchmark(str(f))['floats'] is True

    def test_detects_float_literal(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("x = 1.5f + 2.0;")
        assert fm.scan_benchmark(str(f))['floats'] is True

    def test_detects_pointer(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("int *ptr = &x;")
        assert fm.scan_benchmark(str(f))['pointers'] is True

    def test_detects_printf(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text('printf("hello %d", n);')
        assert fm.scan_benchmark(str(f))['strings'] is True

    def test_no_features_in_empty_file(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("")
        features = fm.scan_benchmark(str(f))
        assert not any(features.values())

    def test_ignores_single_line_comments(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("// for (int i = 0; i < n; i++) {}")
        assert fm.scan_benchmark(str(f))['loops'] is False

    def test_ignores_block_comments(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("/* for (int i = 0; i < n; i++) {} */")
        assert fm.scan_benchmark(str(f))['loops'] is False

    def test_returns_all_expected_feature_keys(self, tmp_path):
        f = tmp_path / "bench.c"
        f.write_text("")
        features = fm.scan_benchmark(str(f))
        expected_keys = {'loops', 'branches', 'functions', 'static_variables',
                         'pointers', 'strings', 'floats'}
        assert set(features.keys()) == expected_keys


# ---------------------------------------------------------------------------
# filter_by_features
# ---------------------------------------------------------------------------

class TestFilterByFeatures:
    def test_keeps_flags_for_detected_features(self):
        categories = {'loops': ['loop-unroll', 'peel-loops'], 'branches': ['tree-if-conversion']}
        features = {'loops': True, 'branches': False}
        result = fm.filter_by_features(categories, features)
        assert 'loop-unroll' in result
        assert 'peel-loops' in result
        assert 'tree-if-conversion' not in result

    def test_all_features_detected(self):
        categories = {'loops': ['loop-unroll'], 'floats': ['float-store']}
        features = {'loops': True, 'floats': True}
        result = fm.filter_by_features(categories, features)
        assert set(result) == {'loop-unroll', 'float-store'}

    def test_no_features_detected_returns_empty(self):
        categories = {'loops': ['loop-unroll'], 'floats': ['float-store']}
        features = {'loops': False, 'floats': False}
        result = fm.filter_by_features(categories, features)
        assert result == []

    def test_category_not_in_features_dict_excluded(self):
        categories = {'unknown_category': ['some-flag']}
        features = {'loops': True}
        result = fm.filter_by_features(categories, features)
        assert result == []

    def test_empty_categories(self):
        result = fm.filter_by_features({}, {'loops': True})
        assert result == []


# ---------------------------------------------------------------------------
# apply_constraints
# ---------------------------------------------------------------------------

class TestApplyConstraints:
    def test_strong_dependency_added(self):
        # forward-propagate requires loop-unroll-and-jam
        result = fm.apply_constraints(['forward-propagate'])
        assert 'loop-unroll-and-jam' in result

    def test_strong_peel_loops_requires_unroll_loops(self):
        result = fm.apply_constraints(['peel-loops'])
        assert 'unroll-loops' in result

    def test_weak_dependency_added(self):
        # sched-interblock works better with schedule-insns
        result = fm.apply_constraints(['sched-interblock'])
        assert 'schedule-insns' in result

    def test_synergistic_gcse_sm_adds_gcse_lm(self):
        result = fm.apply_constraints(['gcse-sm'])
        assert 'gcse-lm' in result

    def test_synergistic_works_both_ways(self):
        result = fm.apply_constraints(['gcse-lm'])
        assert 'gcse-sm' in result

    def test_excluded_flag_not_added(self):
        # peel-loops has weak dep on auto-profile, but auto-profile is excluded
        # (peel-loops also needs unroll-loops via strong dep, so include that too)
        result = fm.apply_constraints(['peel-loops', 'unroll-loops'])
        assert 'auto-profile' not in result

    def test_unrelated_flags_unchanged(self):
        flags = ['some-random-flag', 'another-flag']
        result = fm.apply_constraints(flags)
        assert 'some-random-flag' in result
        assert 'another-flag' in result

    def test_result_is_sorted(self):
        flags = ['zzz-flag', 'aaa-flag', 'mmm-flag']
        result = fm.apply_constraints(flags)
        assert result == sorted(result)

    def test_no_duplicates_when_pair_already_present(self):
        # Both gcse-sm and gcse-lm already in — synergistic constraint satisfied
        result = fm.apply_constraints(['gcse-sm', 'gcse-lm'])
        assert result.count('gcse-sm') == 1
        assert result.count('gcse-lm') == 1

    def test_chained_dependencies_all_resolved(self):
        # forward-propagate → loop-unroll-and-jam (strong)
        # peel-loops → unroll-loops (strong)
        result = fm.apply_constraints(['forward-propagate', 'peel-loops'])
        assert 'loop-unroll-and-jam' in result
        assert 'unroll-loops' in result

    def test_empty_input_returns_empty(self):
        result = fm.apply_constraints([])
        assert result == []

    def test_modulo_sched_allow_regmoves_requires_modulo_sched(self):
        result = fm.apply_constraints(['modulo-sched-allow-regmoves'])
        assert 'modulo-sched' in result

    def test_ipa_vrp_requires_ipa_cp(self):
        result = fm.apply_constraints(['ipa-vrp'])
        assert 'ipa-cp' in result

    def test_strong_excluded_dependency_skipped(self, monkeypatch):
        """If a strong dependency is in EXCLUDED_FLAGS it should be skipped, not added."""
        monkeypatch.setitem(
            fm.PDCAT_CONSTRAINTS, 'strong',
            [('forward-propagate', 'auto-profile')]  # auto-profile is excluded
        )
        result = fm.apply_constraints(['forward-propagate'])
        assert 'auto-profile' not in result

    def test_synergistic_excluded_partner_a_skipped(self, monkeypatch):
        """Synergistic: if flag_b present but flag_a is excluded, flag_a must not be added."""
        monkeypatch.setitem(
            fm.PDCAT_CONSTRAINTS, 'synergistic',
            [('auto-profile', 'gcse-sm')]  # auto-profile is excluded
        )
        result = fm.apply_constraints(['gcse-sm'])
        assert 'auto-profile' not in result

    def test_synergistic_excluded_partner_b_skipped(self, monkeypatch):
        """Synergistic: if flag_a present but flag_b is excluded, flag_b must not be added."""
        monkeypatch.setitem(
            fm.PDCAT_CONSTRAINTS, 'synergistic',
            [('gcse-sm', 'auto-profile')]  # auto-profile is excluded
        )
        result = fm.apply_constraints(['gcse-sm'])
        assert 'auto-profile' not in result


# ---------------------------------------------------------------------------
# build_flag_list (full pipeline)
# ---------------------------------------------------------------------------

class TestBuildFlagList:
    def _make_cfsca_file(self, tmp_path, content):
        f = tmp_path / "CFSCA_flags.txt"
        f.write_text(content)
        return str(f)

    def _make_benchmark(self, tmp_path, content):
        f = tmp_path / "bench.c"
        f.write_text(content)
        return str(f)

    def test_returns_list_of_strings(self, tmp_path):
        cfsca = self._make_cfsca_file(tmp_path, "# loops\nloop-unroll\npeel-loops\n")
        bench = self._make_benchmark(tmp_path, "for (int i=0; i<n; i++) {}")
        result = fm.build_flag_list(cfsca, bench)
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)

    def test_includes_feature_matched_flags(self, tmp_path):
        cfsca = self._make_cfsca_file(tmp_path, "# loops\nloop-unroll\n# floats\nfloat-store\n")
        bench = self._make_benchmark(tmp_path, "for (int i=0; i<n; i++) {}")
        result = fm.build_flag_list(cfsca, bench)
        assert 'loop-unroll' in result
        assert 'float-store' not in result  # no floats in source

    def test_excludes_undetected_feature_flags(self, tmp_path):
        cfsca = self._make_cfsca_file(tmp_path, "# floats\nfloat-store\n")
        bench = self._make_benchmark(tmp_path, "int x = 1;")  # no floats
        result = fm.build_flag_list(cfsca, bench)
        assert result == []

    def test_constraints_applied_in_pipeline(self, tmp_path):
        """forward-propagate should pull in loop-unroll-and-jam via strong constraint."""
        cfsca = self._make_cfsca_file(tmp_path, "# loops\nforward-propagate\n")
        bench = self._make_benchmark(tmp_path, "for (int i=0; i<n; i++) {}")
        result = fm.build_flag_list(cfsca, bench)
        assert 'forward-propagate' in result
        assert 'loop-unroll-and-jam' in result

    def test_result_is_sorted(self, tmp_path):
        cfsca = self._make_cfsca_file(tmp_path, "# loops\nzzz-flag\naaa-flag\n")
        bench = self._make_benchmark(tmp_path, "for (int i=0; i<n; i++) {}")
        result = fm.build_flag_list(cfsca, bench)
        assert result == sorted(result)
