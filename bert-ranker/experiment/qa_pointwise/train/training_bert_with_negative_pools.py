import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn import BCELoss

from experiment.bert_utils import BertTraining
from experiment.bert_utils import InputExample, convert_examples_to_features


class BertTrainingPointwiseWithNegativePools(BertTraining):
    def __init__(self, *args, **kwargs):
        """This does random sampling of negative answers. Optionally, if n_train_answers is set to a value greater than
        one, this will sample for each question *n_train_answers* negative answers, which are ranked with the currently
        trained model. The best-ranked answer is chosen as a negative sample.

        :param args:
        :param kwargs:
        """
        super(BertTrainingPointwiseWithNegativePools, self).__init__(*args, **kwargs)
        self.cur_dataset = 0
        self.n_batches = None

        self.max_negative_per_answer = self.config.get('max_negative_per_answer', 10)
        self.negative_sample_from_top_n = self.config.get('negative_sample_from_top_n', False)
        # negative samples can be randomly sampled from the top-n of a pool. Sometimes the pools position indicates
        # similarity from the retrieval (e.g., InsuranceQA v2). However, for some datasets the most similar might be too
        # similar to train a model. Thus we enable the option to draw negative samples from the top-n

        # self.max_negative_per_answer = self.config.get('neg', 10)
        self.random_flip_inputs = self.config.get('random_flip_inputs', False)
        # Randomly flip inputs (titles, bodies) 50% of the time during training

        self.train_instances = defaultdict(lambda: list())
        self.train_instances_epoch_index = dict()

    def potentially_flip_inputs(self, a, b):
        if self.random_flip_inputs:
            if bool(random.getrandbits(1)):
                return a,b
            else:
                return b, a
        else:
            return a,b

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
        super(BertTrainingPointwiseWithNegativePools, self).prepare_training(model, data)

        # create features for all training examples
        for i, dataset in enumerate(data.datas_train):
            dataset_examples = []
            for pool in dataset.archive.train.qa:
                for a in pool.ground_truth:
                    input_a, input_b = self.potentially_flip_inputs(pool.question.text, a.text)
                    dataset_examples.append(
                        InputExample(pool.question.metadata['id'], input_a, input_b, label=1.0)
                    )
                    
                neg = []

                pot_neg_answer_list = pool.pooled_answers
                if self.negative_sample_from_top_n is not False:
                    pot_neg_answer_list = list(pool.pooled_answers[:self.negative_sample_from_top_n])
                    random.shuffle(pot_neg_answer_list)

                for a in pot_neg_answer_list:
                    if a in pool.ground_truth:
                        continue

                    input_a, input_b = self.potentially_flip_inputs(pool.question.text, a.text)
                    neg.append(
                        InputExample(pool.question.metadata['id'], input_a, input_b, label=0.0)
                    )
                    if len(neg) >= self.max_negative_per_answer * len(pool.ground_truth):
                        break
                dataset_examples += neg
                    
            self.train_instances[i] = convert_examples_to_features(
                dataset_examples, self.length_question + self.length_answer, model.tokenizer, self.logger
            )

        # calculate number of batches
        min_examples = min([len(v) for v in self.train_instances.values()])
        n_datasets = len(self.train_instances)
        if self.epoch_max_examples is None or min_examples * n_datasets < self.epoch_max_examples:
            examples_per_epoch = min_examples * n_datasets
        else:
            examples_per_epoch = self.epoch_max_examples
        self.n_batches = math.ceil(examples_per_epoch / float(self.batchsize))

    def prepare_next_epoch(self, model, data, epoch):
        super(BertTrainingPointwiseWithNegativePools, self).prepare_next_epoch(model, data, epoch)

        # we shuffle the train instances for each epoch so that there is no need to
        for dataset, instances in self.train_instances.items():
            random.shuffle(instances)
            self.train_instances_epoch_index[dataset] = 0

    def get_n_batches(self):
        return self.n_batches

    def get_next_batch(self, model, data):
        self.cur_dataset = (self.cur_dataset + 1) % len(self.train_instances)

        # indices
        index_start = self.train_instances_epoch_index[self.cur_dataset]
        index_end = index_start + self.batchsize

        batch_examples = self.train_instances[self.cur_dataset][index_start:index_end]

        # set the new dataset index and batch index
        self.train_instances_epoch_index[self.cur_dataset] = index_end
        self.batch_i += 1

        return batch_examples


component = BertTrainingPointwiseWithNegativePools
