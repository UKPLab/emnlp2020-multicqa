import os
import shutil

import torch


class QATrainStatePyTorch(object):
    def __init__(self, path, less_is_better, logger):
        """See experiment.qa_pointwise.train.QATrainState"""
        self.path = path
        self.logger = logger
        self.less_is_better = less_is_better

        self.initialize()

    def initialize(self):
        self.scores = []
        self.best_score = -1 if not self.less_is_better else 2
        self.best_epoch = 0
        self.recorded_epochs = 0
        if self.path and not os.path.exists(self.path):
            os.mkdir(self.path)

    def load(self, model, optimizer, weights='last', strict=True):
        """
        :param weights: 'last' or 'best'
        :return:
        """
        if os.path.exists(self.scores_file):
            scores = []
            with open(self.scores_file, 'r') as f:
                for line in f:
                    scores.append(float(line))

            self.scores = scores
            op = max if not self.less_is_better else min
            self.best_score = op(scores)
            self.best_epoch = scores.index(self.best_score) + 1
            self.recorded_epochs = len(scores)

        if isinstance(weights, str):
            restore_path = self.checkpoint_file(self.recorded_epochs if weights == 'last' else self.best_epoch)
        else:
            restore_path = self.checkpoint_file(weights)

        if os.path.exists(restore_path):
            self.logger.info('Restoring model weights from: {}'.format(restore_path))
            checkpoint = torch.load(restore_path)
            model.load_state_dict(checkpoint['model'], strict=strict)
            if optimizer is not None and checkpoint.get('optimizer') is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.logger.info('Could not restore weights. Path does not exist: {}'.format(restore_path))

    def record(self, model, optimizer, score, backup_checkpoint_every=-1):
        """Records a checkpoint

        :param model:
        :param optimizer:
        :param score:
        :param backup_checkpoint_every: optionally we can set a value to keep the checkpoint of every n-th epoch. This
                is used for measuring performance with varying training times later
        :return:
        """
        self.recorded_epochs += 1
        self.scores.append(score)
        with open(self.scores_file, 'a') as f:
            f.write('{}\n'.format(score))

        self.logger.info('Checkpointing epoch {}'.format(self.recorded_epochs))

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None
        }, self.checkpoint_file(self.recorded_epochs))
        model.config.to_json_file(os.path.join(self.path, "huggingface-config.json"))

        if (not self.less_is_better and score > self.best_score) or (self.less_is_better and score < self.best_score):
            self.logger.info('New best score {} (previous: {})'.format(score, self.best_score))
            self.best_score = score
            self.best_epoch = self.recorded_epochs

        # remove old checkpoints
        for i in range(1, self.recorded_epochs):
            if backup_checkpoint_every > 0 and i % backup_checkpoint_every == 0:
                continue

            save_path = self.checkpoint_file(i)
            if i != self.best_epoch and os.path.exists(save_path):
                os.remove(save_path)

    def clear(self):
        shutil.rmtree(self.path)
        self.initialize()

    @property
    def scores_file(self):
        return os.path.join(self.path, 'scores.txt')

    def checkpoint_file(self, epoch):
        return os.path.join(self.path, 'model-checkpoint-{}.tar'.format(epoch))
