import importlib
import os
from os import path

import click
import numpy as np
import tensorflow as tf

from experiment import MultiData
from experiment.checkpointing import QATrainStatePyTorch
from experiment.config import load_config, write_config
from experiment.run_util import setup_logger


@click.command()
@click.argument('config_file')
@click.option('--do-test/--no-do-test', default=False)
def run_cmd(config_file, do_test):
    config_eval = load_config(config_file)
    config_experiment = load_config(config_eval['global']['experiment_config'])

    def run():
        """This program is the starting point for evaluations without training"""
        run_name = config_eval.get('run_name', 'undefined name')
        config_eval_global = config_eval['global']
        config_experiment_global = config_experiment['global']

        output_dir = config_eval_global['evaluation_output']
        restore_weights = config_eval_global.get('restore_epoch', 'best')

        if os.path.exists(path.join(output_dir, 'log.txt')):
            with open(path.join(output_dir, 'log.txt'), 'r') as f:
                if 'Results for all datasets in dev' in f.read():
                    return 'exp. already finished'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        write_config(config_eval, output_dir)

        # setup a logger
        logger = setup_logger(config_eval['logger'], output_dir, name='experiment')

        # we allow to set the random seed in the config file for reproducibility. However, when running on GPU, results
        # can still be nondeterministic
        if 'random_seed' in config_experiment_global:
            seed = config_experiment_global['random_seed']
            logger.info('Using fixed random seed'.format(seed))
            np.random.seed(seed)
            tf.set_random_seed(seed)

        multi_data_module_eval = MultiData(config_eval['data'], config_eval_global, logger)
        for dataset_config in config_eval['data']:
            data_module = dataset_config['data-module']
            splits = dataset_config.get('splits', ['train', 'dev', 'test'])
            DataClass = importlib.import_module(data_module).component
            data = DataClass(dataset_config, config_eval_global, logger)
            multi_data_module_eval.add(data, splits)

        model_module = config_experiment['model-module']
        evaluation_module = config_eval['evaluation-module']

        # The modules are now dynamically loaded
        ModelClass = importlib.import_module(model_module).component
        EvaluationClass = importlib.import_module(evaluation_module).component

        # We then wire together all the modules and start training
        model = ModelClass(config_experiment['model'], config_experiment_global, logger)
        evaluation = EvaluationClass(config_eval['evaluation'], config_eval_global, logger)

        # setup the data (validate, create generators, load data, or else)
        logger.info('Setting up the data')
        multi_data_module_eval.setup()
        # build the model (e.g. compile it)
        logger.info('Building the model')

        model.set_bert_configs_from_output_folder(restore_weights)     # sets necessary huggingface bert restore paths
        model.build(multi_data_module_eval)
        if hasattr(model.bert.config, 'adapter_attention'):
            model.bert.bert.enable_adapters(unfreeze_adapters=True, unfreeze_attention=True)
        elif hasattr(model.bert.config, 'adapters'):
            model.bert.bert.enable_adapters(unfreeze_adapters=True, unfreeze_attention=False)

        state = QATrainStatePyTorch(path.join(config_experiment_global['output_path'], 'checkpoints'),
                                    less_is_better=False, logger=logger)
        state.load(model.bert, optimizer=None, weights=restore_weights, strict=False)

        logger.info('Evaluating')
        results = evaluation.start(model, multi_data_module_eval, valid_only=not do_test)

        logger.info('DONE')

        return {
            'run_name': run_name,
            'results': results
        }

    run()


if __name__ == '__main__':
    run_cmd()
