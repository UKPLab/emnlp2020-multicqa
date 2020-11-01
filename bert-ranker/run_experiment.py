import importlib
import os

import click
import numpy as np
import tensorflow as tf

from experiment import MultiData
from experiment.config import load_config, write_config
from experiment.run_util import setup_logger


@click.command()
@click.argument('config_file')
def run_cmd(config_file):
    config = load_config(config_file)

    def run():
        """This program is the starting point for every experiment. It pulls together the configuration and all
        necessary experiment classes to load

        """
        run_name = config.get('run_name', 'undefined name')
        config_global = config['global']

        if not os.path.exists(config_global['output_path']):
            os.mkdir(config_global['output_path'])

        write_config(config, config_global['output_path'])

        # setup a logger
        logger = setup_logger(config['logger'], config_global['output_path'], name='experiment')

        # we allow to set the random seed in the config file for reproducibility. However, when running on GPU, results
        # can still be nondeterministic
        if 'random_seed' in config_global:
            seed = config_global['random_seed']
            logger.info('Using fixed random seed'.format(seed))
            np.random.seed(seed)
            tf.set_random_seed(seed)

        # We are now fetching all relevant modules. It is strictly required that these module contain a variable named
        # 'component' that points to a class which inherits from experiment.Data, experiment.Experiment,
        # experiment.Trainer or experiment.Evaluator
        multi_data_module = MultiData(config['data'], config, logger)
        for dataset_config in config['data']:
            data_module = dataset_config['data-module']
            splits = dataset_config.get('splits', ['train', 'dev', 'test'])
            DataClass = importlib.import_module(data_module).component
            data = DataClass(dataset_config, config_global, logger)
            multi_data_module.add(data, splits)

        model_module = config['model-module']
        training_module = config['training-module']
        evaluation_module = config.get('evaluation-module', None)

        # The modules are now dynamically loaded
        ModelClass = importlib.import_module(model_module).component
        TrainingClass = importlib.import_module(training_module).component
        EvaluationClass = importlib.import_module(evaluation_module).component

        if "adapter_mean_estimation-module" in config:
            ame_module = config["adapter_mean_estimation-module"]
            AMEClass = importlib.import_module(ame_module).component
            ame = AMEClass(config['adapter_mean_estimation'], config_global, logger)

        # We then wire together all the modules and start training
        model = ModelClass(config['model'], config_global, logger)
        training = TrainingClass(config['training'], config_global, logger)
        evaluation = EvaluationClass(config['evaluation'], config_global, logger)

        # setup the data (validate, create generators, load data, or else)
        logger.info('Setting up the data')
        multi_data_module.setup()
        # build the model (e.g. compile it)
        logger.info('Building the model')
        model.build(multi_data_module)

        # start the training process
        if not config["training"].get("skip", False):
            logger.info('Starting the training process')
            training.start(model, multi_data_module, evaluation)
        dev_score = training.restore_best_epoch(model, multi_data_module, evaluation)

        # perform evaluation, if required
        if not config['evaluation'].get('skip', False):
            logger.info('Evaluating')
            results = evaluation.start(model, multi_data_module, valid_only=False)
        else:
            logger.info('Skipping evaluation')
            results = {'dev': dev_score}

        logger.info('DONE')

        return {
            'run_name': run_name,
            'results': results
        }

    run()


if __name__ == '__main__':
    run_cmd()
