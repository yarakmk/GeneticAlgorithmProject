import opentuner
from opentuner import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter, FloatParameter, EnumParameter
from opentuner.measurement import MeasurementInterface
import geneticAlgorithm as ga  # ← import your GA as a module

TUNING_BENCHMARK = 'PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c'
TUNING_POLYBENCH = True
TUNING_EXTRA_SOURCES  = ['PolyBenchC-4.2.1/utilities/polybench.c']
TUNING_DEFINES = ['POLYBENCH_TIME', 'STANDARD_DATASET']
TUNING_EXTRA_INCLUDES = [
    'PolyBenchC-4.2.1/utilities',
    'PolyBenchC-4.2.1/linear-algebra/blas/gemm'
]
class GAHyperparamTuner(MeasurementInterface):

    def manipulator(self):
        m = ConfigurationManipulator()

        # Categorical — OpenTuner picks one string from the list
        m.add_parameter(EnumParameter('crossover_type', [
            'one_point', 'two_point', 'uniform', 'shuffle', 'segment'
        ]))
        m.add_parameter(EnumParameter('mutation_type', [
            'bit_flip', 'gauss_by_center', 'uniform_mutation'
        ]))
        m.add_parameter(EnumParameter('selection_type', [
            'fully_random', 'roulette', 'stochastic',
            'sigma_scaling', 'ranking', 'linear_ranking', 'tournament'
        ]))

        # Numerical — OpenTuner picks a value within the range
        m.add_parameter(IntegerParameter('population_size', 20, 100))
        m.add_parameter(FloatParameter('crossover_rate', 0.05, 0.5))
        m.add_parameter(FloatParameter('mutation_rate', 0.01, 0.5))
        m.add_parameter(FloatParameter('elitism_ratio', 0.05, 0.4))
        m.add_parameter(FloatParameter('parents_portion', 0.3, 0.9))
        m.add_parameter(IntegerParameter('max_generations', 20, 50))
        m.add_parameter(IntegerParameter('max_no_improvement', 10, 25))

        return m

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data

        ga.CROSSOVER_TYPE     = cfg['crossover_type']
        ga.MUTATION_TYPE      = cfg['mutation_type']
        ga.SELECTION_TYPE     = cfg['selection_type']
        ga.POPULATION_SIZE    = cfg['population_size']
        ga.CROSSOVER_RATE     = cfg['crossover_rate']
        ga.MUTATION_RATE      = cfg['mutation_rate']
        ga.ELITISM_RATIO      = cfg['elitism_ratio']
        ga.PARENTS_PORTION    = cfg['parents_portion']
        ga.MAX_GENERATIONS    = cfg['max_generations']
        ga.MAX_NO_IMPROVEMENT = cfg['max_no_improvement']

        result = ga.run_genetic_algorithm(
            TUNING_BENCHMARK,
            polybench=TUNING_POLYBENCH,
            extra_sources=TUNING_EXTRA_SOURCES,
            extra_includes=TUNING_EXTRA_INCLUDES,
            defines=TUNING_DEFINES
        )
        return opentuner.resultsdb.models.Result(time=result['best_fitness'])

# if __name__ == '__main__':
#     args = opentuner.default_argparser().parse_args()
#     GAHyperparamTuner.main(args)