search_space = {
    'filter_count_factor': {'_type': 'choice', '_value': [1, 2, 4, 6, 8, 12, 16]},
    'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
}

from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python hyperparameter_opt.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2

experiment.run(8080)

input()
experiment.stop()