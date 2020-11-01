import math

import numpy as np
import torch
from torch.nn import BCELoss

from experiment.bert_utils import BertTraining
from experiment.bert_utils import InputExample, convert_examples_to_features


class BertTrainingPointwiseAllQSameBatch(BertTraining):
    def __init__(self, *args, **kwargs):
        """This does random sampling of negative answers. Optionally, if n_train_answers is set to a value greater than
        one, this will sample for each question *n_train_answers* negative answers, which are ranked with the currently
        trained model. The best-ranked answer is chosen as a negative sample.

        :param args:
        :param kwargs:
        """
        super(BertTrainingPointwiseAllQSameBatch, self).__init__(*args, **kwargs)
        self.cur_dataset = 0
        self.n_batches = None

    def get_loss(self, model, step_examples, tasks=None):
        """
        :param model: BertWrapperModel
        :param step_examples: list[InputExample]
        :param tasks: list[str] or None of adapter name(s)
        :return:
        """
        model.bert.train()

        input_ids = torch.tensor([f.input_ids for f in step_examples], dtype=torch.long).to(model.device)
        input_mask = torch.tensor([f.input_mask for f in step_examples], dtype=torch.long).to(model.device)
        segment_ids = torch.tensor([f.segment_ids for f in step_examples], dtype=torch.long).to(
            model.device)
        label_ids = torch.tensor([f.label_id for f in step_examples], dtype=torch.float).to(model.device)

        loss_fct = BCELoss()
        step_loss = loss_fct(model.bert(input_ids, segment_ids, input_mask, tasks=tasks), label_ids.view(-1, 1))

        return step_loss

    def prepare_training(self, model, data):
        super(BertTrainingPointwiseAllQSameBatch, self).prepare_training(model, data)

        # create features for all training examples
        for i, dataset in enumerate(data.datas_train):
            dataset_examples = []
            for pool in dataset.archive.train.qa:
                for gt in pool.ground_truth:
                    self.train_questions[i].append(pool.question)
                    dataset_examples.append(
                        InputExample(pool.question.metadata['id'], pool.question.text, gt.text, label=1.0)
                    )
            self.train_positive_instances[i] = convert_examples_to_features(
                dataset_examples, self.length_question + self.length_answer, model.tokenizer, self.logger
            )

        # calculate number of batches
        min_examples = min([len(v) for v in self.train_questions.values()])
        n_datasets = len(self.train_questions)
        if self.epoch_max_examples is None or min_examples * n_datasets < self.epoch_max_examples:
            examples_per_epoch = min_examples * n_datasets
        else:
            examples_per_epoch = self.epoch_max_examples
        self.n_batches = examples_per_epoch

    def get_n_batches(self):
        return self.n_batches

    def get_next_batch(self, model, data):
        self.cur_dataset = (self.cur_dataset + 1) % len(self.train_questions)
        n_positive = self.batchsize // self.n_train_answers
        example_index = self.get_batch_indices(self.cur_dataset, n_positive)

        batch_examples = []
        for i in example_index:
            unrelated_answers = [
                InputExample(self.train_questions[self.cur_dataset][i].metadata['id'],
                             self.train_questions[self.cur_dataset][i].text,
                             a.text, label=0.0)
                for a in self.get_random_answers(self.cur_dataset, self.n_train_answers - 1)
            ]
            unrelated_answers_features = convert_examples_to_features(
                unrelated_answers, self.length_question + self.length_answer, model.tokenizer, self.logger,
                show_example=False
            )
            batch_examples += [self.train_positive_instances[self.cur_dataset][i]] + unrelated_answers_features

        self.batch_i += n_positive
        return batch_examples


component = BertTrainingPointwiseAllQSameBatch
