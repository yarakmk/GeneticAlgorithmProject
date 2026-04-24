"""
Unit tests for hyperparameter_tuner.py.

opentuner is available, so we import it directly.
ga.run_genetic_algorithm is mocked so no compiler or benchmark is needed.
"""
import pytest
import runpy
from unittest.mock import patch, MagicMock
import algorithm.geneticAlgorithm as ga
import algorithm.hyperparameter_tuner as ht
from algorithm.hyperparameter_tuner import GAHyperparamTuner
from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter, FloatParameter, EnumParameter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tuner():
    """Instantiate GAHyperparamTuner without OpenTuner's full session setup."""
    with patch('opentuner.measurement.MeasurementInterface.__init__', return_value=None):
        tuner = GAHyperparamTuner.__new__(GAHyperparamTuner)
        GAHyperparamTuner.__init__(tuner)
    return tuner


def make_desired_result(cfg: dict):
    """Build a fake desired_result object with configuration.data = cfg."""
    dr = MagicMock()
    dr.configuration.data = cfg
    return dr


SAMPLE_CFG = {
    'crossover_type':    'one_point',
    'mutation_type':     'bit_flip',
    'selection_type':    'linear_ranking',
    'population_size':   40,
    'crossover_rate':    0.35,
    'mutation_rate':     0.15,
    'elitism_ratio':     0.2,
    'parents_portion':   0.5,
    'max_generations':   25,
    'max_no_improvement': 12,
}


# ---------------------------------------------------------------------------
# manipulator()
# ---------------------------------------------------------------------------

class TestManipulator:
    def setup_method(self):
        self.tuner = make_tuner()
        self.m = self.tuner.manipulator()

    def test_returns_configuration_manipulator(self):
        assert isinstance(self.m, ConfigurationManipulator)

    def test_has_all_expected_parameters(self):
        names = {p.name for p in self.m.params}
        expected = {
            'crossover_type', 'mutation_type', 'selection_type',
            'population_size', 'crossover_rate', 'mutation_rate',
            'elitism_ratio', 'parents_portion', 'max_generations',
            'max_no_improvement',
        }
        assert expected == names

    def test_crossover_type_is_enum(self):
        param = next(p for p in self.m.params if p.name == 'crossover_type')
        assert isinstance(param, EnumParameter)

    def test_crossover_type_options(self):
        param = next(p for p in self.m.params if p.name == 'crossover_type')
        assert set(param.options) == {'one_point', 'two_point', 'uniform', 'shuffle', 'segment'}

    def test_mutation_type_is_enum(self):
        param = next(p for p in self.m.params if p.name == 'mutation_type')
        assert isinstance(param, EnumParameter)

    def test_selection_type_is_enum(self):
        param = next(p for p in self.m.params if p.name == 'selection_type')
        assert isinstance(param, EnumParameter)

    def test_population_size_is_integer(self):
        param = next(p for p in self.m.params if p.name == 'population_size')
        assert isinstance(param, IntegerParameter)

    def test_crossover_rate_is_float(self):
        param = next(p for p in self.m.params if p.name == 'crossover_rate')
        assert isinstance(param, FloatParameter)

    def test_mutation_rate_is_float(self):
        param = next(p for p in self.m.params if p.name == 'mutation_rate')
        assert isinstance(param, FloatParameter)

    def test_population_size_bounds(self):
        param = next(p for p in self.m.params if p.name == 'population_size')
        assert param.min_value == 20
        assert param.max_value == 100

    def test_max_generations_bounds(self):
        param = next(p for p in self.m.params if p.name == 'max_generations')
        assert param.min_value == 20
        assert param.max_value == 50


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestRun:
    def setup_method(self):
        self.tuner = make_tuner()

    def _run_with_cfg(self, cfg=None):
        cfg = cfg or SAMPLE_CFG
        dr = make_desired_result(cfg)
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value={'best_fitness': 0.42, 'best_chromosome': [], 'best_flags': [], 'log': []}):
            return self.tuner.run(dr, input=None, limit=None)

    def test_returns_result_object(self):
        import opentuner.resultsdb.models as models
        result = self._run_with_cfg()
        assert isinstance(result, models.Result)

    def test_result_time_equals_best_fitness(self):
        result = self._run_with_cfg()
        assert result.time == pytest.approx(0.42)

    def test_sets_ga_crossover_type(self):
        self._run_with_cfg()
        assert ga.CROSSOVER_TYPE == SAMPLE_CFG['crossover_type']

    def test_sets_ga_mutation_type(self):
        self._run_with_cfg()
        assert ga.MUTATION_TYPE == SAMPLE_CFG['mutation_type']

    def test_sets_ga_selection_type(self):
        self._run_with_cfg()
        assert ga.SELECTION_TYPE == SAMPLE_CFG['selection_type']

    def test_sets_ga_population_size(self):
        self._run_with_cfg()
        assert ga.POPULATION_SIZE == SAMPLE_CFG['population_size']

    def test_sets_ga_crossover_rate(self):
        self._run_with_cfg()
        assert ga.CROSSOVER_RATE == pytest.approx(SAMPLE_CFG['crossover_rate'])

    def test_sets_ga_mutation_rate(self):
        self._run_with_cfg()
        assert ga.MUTATION_RATE == pytest.approx(SAMPLE_CFG['mutation_rate'])

    def test_sets_ga_elitism_ratio(self):
        self._run_with_cfg()
        assert ga.ELITISM_RATIO == pytest.approx(SAMPLE_CFG['elitism_ratio'])

    def test_sets_ga_parents_portion(self):
        self._run_with_cfg()
        assert ga.PARENTS_PORTION == pytest.approx(SAMPLE_CFG['parents_portion'])

    def test_sets_ga_max_generations(self):
        self._run_with_cfg()
        assert ga.MAX_GENERATIONS == SAMPLE_CFG['max_generations']

    def test_sets_ga_max_no_improvement(self):
        self._run_with_cfg()
        assert ga.MAX_NO_IMPROVEMENT == SAMPLE_CFG['max_no_improvement']

    def test_calls_run_genetic_algorithm_with_tuning_benchmark(self):
        dr = make_desired_result(SAMPLE_CFG)
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value={'best_fitness': 0.5, 'best_chromosome': [], 'best_flags': [], 'log': []}) as mock_ga:
            self.tuner.run(dr, input=None, limit=None)
        call_args = mock_ga.call_args
        assert call_args[0][0] == ht.TUNING_BENCHMARK

    def test_passes_polybench_flag(self):
        dr = make_desired_result(SAMPLE_CFG)
        with patch('algorithm.geneticAlgorithm.run_genetic_algorithm',
                   return_value={'best_fitness': 0.5, 'best_chromosome': [], 'best_flags': [], 'log': []}) as mock_ga:
            self.tuner.run(dr, input=None, limit=None)
        assert mock_ga.call_args[1]['polybench'] == ht.TUNING_POLYBENCH

