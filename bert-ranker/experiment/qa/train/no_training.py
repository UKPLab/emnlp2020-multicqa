import tensorflow as tf

import experiment


class NoTraining(experiment.Training):
    """This is a replacement component that skips the training process"""

    def __init__(self, config, config_global, logger):
        super(NoTraining, self).__init__(config, config_global, logger)

    def start(self, model, data, evaluation):
        self.logger.info("Skipping training")

    def restore_best_epoch(self, model, data, evaluation):
        pass

    def remove_checkpoints(self):
        pass


component = NoTraining
