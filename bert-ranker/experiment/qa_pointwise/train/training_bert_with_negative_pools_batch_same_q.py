import math
from collections import defaultdict

import numpy as np
import torch
from torch.nn import BCELoss

from experiment.bert_utils import BertTraining
from experiment.bert_utils import InputExample, convert_examples_to_features


class BertTrainingPointwiseWithNegativePoolsAllQSameBatch(BertTraining):
    def __init__(self, *args, **kwargs):
        """Same as training_bert_batch_same_q but every batch contains pos/neg examples from the current question

        :param args:
        :param kwargs:
        """
        super(BertTrainingPointwiseWithNegativePoolsAllQSameBatch, self).__init__(*args, **kwargs)
        self.cur_dataset = 0
        self.n_batches = None

        self.batches = defaultdict(lambda: list())

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
        super(BertTrainingPointwiseWithNegativePoolsAllQSameBatch, self).prepare_training(model, data)

        # create features for all training examples
        for i, dataset in enumerate(data.datas_train):
            dataset_batches = []
            for pool in dataset.archive.train.qa:
                batch = []
                for a in pool.pooled_answers:
                    batch.append(
                        InputExample(pool.question.metadata['id'], pool.question.text, a.text,
                                     label=1.0 if a in pool.ground_truth else 0.0)
                    )
                dataset_batches.append(
                    convert_examples_to_features(
                        batch, self.length_question + self.length_answer, model.tokenizer, self.logger,
                        show_example=False
                    )
                )
            self.batches[i] = dataset_batches

        # calculate number of batches
        min_batches = min([len(v) for v in self.batches.values()])
        self.n_batches = min_batches * len(self.batches)

    def get_n_batches(self):
        return self.n_batches

    def get_batch_indices(self, dataset_id, size):
        indices = self.train_random_indices[dataset_id]
        batch_indices = indices[:size]
        missing_indices = size - len(batch_indices)

        if missing_indices > 0:
            indices = np.random.permutation(len(self.batches[dataset_id])).tolist()
            batch_indices += indices[:missing_indices]
            remaining_indices = indices[missing_indices:]
        else:
            remaining_indices = indices[size:]

        self.train_random_indices[dataset_id] = remaining_indices
        return batch_indices

    def get_next_batch(self, model, data):
        self.cur_dataset = (self.cur_dataset + 1) % len(self.batches)
        batch_index = self.get_batch_indices(self.cur_dataset, size=1)[0]

        batch_examples = self.batches[self.cur_dataset][batch_index]

        self.batch_i += len(batch_examples)
        return batch_examples


component = BertTrainingPointwiseWithNegativePoolsAllQSameBatch
