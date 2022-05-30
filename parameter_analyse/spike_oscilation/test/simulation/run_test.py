from parameter_analyse.spike_oscilation.python_file.run.run_exploration import run_exploration_2D
from parameter_analyse.spike_oscilation.python_file.parameters import parameter_default


path='/home/kusch/Documents/project/Zerlaut/compare_zerlaut/parameter_analyse/spike_oscilation/simulation/simulation/'

run_exploration_2D(path, parameter_default, 'test.db', 'test_1',
                   { 'frequency':[10.0, 20.0], 'amplitude':[2000.0, 8000.0]},
                   0.0, 5000.0, analyse=True, simulation=True)