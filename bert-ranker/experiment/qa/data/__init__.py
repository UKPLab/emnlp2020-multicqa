import numpy as np

import experiment
from experiment.qa.data import models


class QAData(experiment.Data):
    def __init__(self, config, config_global, logger):
        super(QAData, self).__init__(config, config_global, logger)

        # public fields
        self.archive = None  # Archive
        self.transfer_archives = []  # list[Archive]
        self.lowercased = self.config.get('lowercased', False)

        self.max_train_samples = self.config.get('max_train_samples')
        self.max_train_samples_offset = self.config.get('max_train_samples_offset', 0)
        # the offset is used for repeated experiments so that we do not use the same small train+dev split again
        # an offset of 1 means that we take the training examples in the range of
        # -- 1 * max_train_samples -> 2 * max_train_samples

        self.max_dev_samples = self.config.get('max_dev_samples')
        self.max_dev_samples_offset = self.config.get('max_dev_samples_offset', 0)

    def _get_train_readers(self):
        """:rtype: list[ArchiveReader]"""
        raise NotImplementedError()

    def _get_transfer_readers(self):
        """:rtype: list[ArchiveReader]"""
        return []

    def setup(self):
        readers = self._get_train_readers()
        self.logger.info('Loading {} train datasets'.format(len(readers)))
        archives = [reader.read() for reader in readers]
        self.archive = archives[0]

        if self.max_train_samples:
            for archive in archives:
                archive.shrink('train', self.max_train_samples, self.max_train_samples_offset)
                print('')
            self.logger.info('Reduced the maximum of training sample of all data archives to a maximum of {}. '
                             'With an offset of {}'.format(
                self.max_train_samples,
                self.max_train_samples_offset
            ))

        if self.max_dev_samples:
            for archive in archives:
                archive.shrink('valid', self.max_dev_samples, self.max_dev_samples_offset)
                print('')
            self.logger.info('Reduced the maximum of dev sample of all data archives to a maximum of {}. '
                             'With an offset of {}'.format(
                self.max_dev_samples,
                self.max_dev_samples_offset
            ))

        if self.config.get('balance_data') is True:
            max_len_train = min([len(a.train.qa) for a in archives])
            max_len_dev = min([len(a.valid.qa) for a in archives])
            for archive in archives:
                archive.shrink('train', max_len_train)
                archive.shrink('valid', max_len_dev)
                # archive.train.qa = archive.train.qa[:max_len_train]
                # archive.valid.qa = archive.valid.qa[:max_len_dev]

            self.logger.info('Balanced all data archives to maximum length for train={}, dev={}'.format(
                max_len_train, max_len_dev
            ))

        for other_archive in archives[1:]:
            self.archive = self.archive.combine(other_archive)
        self.logger.debug('Train dataset questions: train={}, dev={}, test={}'.format(
            len(self.archive.train.qa),
            len(self.archive.valid.qa),
            [len(t.qa) for t in self.archive.test]
        ))

        qas = self.archive.train.qa + self.archive.valid.qa
        for t in self.archive.test:
            qas += t.qa
        self.logger.debug('Mean answer count={}'.format(
            np.mean([len(p.ground_truth) for p in qas])
        ))
        self.logger.debug('Mean poolsize={}'.format(
            np.mean([len(p.pooled_answers) for p in qas if p.pooled_answers is not None])
        ))

        self.transfer_archives = [r.read() for r in self._get_transfer_readers()]
        if self.transfer_archives:
            self.logger.debug('Transfer datasets with test questions: {}'.format(
                ', '.join(['{}={}'.format(a.name, [len(t.qa) for t in a.test]) for a in self.transfer_archives])
            ))
