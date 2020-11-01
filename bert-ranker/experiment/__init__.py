from collections import OrderedDict
from os import path

import progressbar


class ComponentBase(object):
    def __init__(self, config, config_global, logger):
        """This is a simple base object for all experiment components

        :type config: dict
        :type config_global: dict
        :type logger: logging.Logger
        """
        self.config = config or dict()
        self.config_global = config_global or dict()
        self.logger = logger

    def create_progress_bar(self, dynamic_msg=None):
        widgets = [
            ' [batch ', progressbar.SimpleProgress(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') '
        ]
        if dynamic_msg is not None:
            widgets.append(progressbar.DynamicMessage(dynamic_msg))

        if self.config_global.get('show_progress', True):
            return progressbar.ProgressBar(widgets=widgets)
        else:
            return progressbar.NullBar()


class Data(ComponentBase):
    def setup(self):
        pass

    def get_fold_data(self, fold_i, n_folds):
        """Generates and returns a new Data instance that contains only the data for a specific fold. This method is
        used for hyperparameter optimization on multiple folds.

        :param fold_i: the number of the current fold
        :param n_folds: the total number of folds
        :return: the data for the specified fold
        """
        raise NotImplementedError()


class MultiData(ComponentBase):
    def __init__(self, config, config_global, logger):
        """MultiData contains multiple Data objects (e.g., to train on more than one dataset, for evaluation etc.)

        :param config:
        :param config_global:
        :param logger:
        """
        super(MultiData, self).__init__(config, config_global, logger)
        self.datas = []
        self._datas_dict = {'train': [], 'dev': [], 'test': []}

    def setup(self):
        for data in self.datas:
            data.setup()

    def add(self, data, splits):
        assert isinstance(splits, list)
        assert len(set(splits) - {'train', 'dev', 'test'}) == 0

        self.datas.append(data)
        for split in splits:
            self._datas_dict[split].append(data)

    @property
    def datas_train(self):
        return self._datas_dict['train']

    @property
    def datas_dev(self):
        return self._datas_dict['dev']

    @property
    def datas_test(self):
        return self._datas_dict['test']


class Model(ComponentBase):
    def __init__(self, config, config_global, logger):
        super(Model, self).__init__(config, config_global, logger)
        self.__summary = None

        self.special_print = OrderedDict()
        # a dictionary to store special tensors that should be printed after each epoch

    def build(self, data):
        raise NotImplementedError()


class Training(ComponentBase):
    def start(self, model, data, evaluation):
        """

        :param model:
        :type model: Model
        :param data:
        :type data: Data
        :type evaluation: Evaluation
        """
        raise NotImplementedError()

    @property
    def checkpoint_folder(self):
        return path.join(self.config_global['output_path'], 'checkpoints')

    @property
    def tensorboard_folder(self):
        return path.join(self.config_global['output_path'], 'tensorboard')

    def restore_best_epoch(self, model, data, evaluation):
        raise NotImplementedError()

    def remove_checkpoints(self):
        """Removes all the persisted checkpoint data that was generated during training for restoring purposes"""
        raise NotImplementedError()


class Evaluation(ComponentBase):
    def start(self, model, data, valid_only=False):
        """

        :type model: Model
        :type data: Data
        :type valid_only: bool
        :return: The score of the primary measure for all tested data splits
        :rtype: dict[basestring, float]
        """
        raise NotImplementedError()