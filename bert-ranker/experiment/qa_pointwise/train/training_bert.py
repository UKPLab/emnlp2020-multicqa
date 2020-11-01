import math

import numpy as np
import torch
from torch.nn import BCELoss

from experiment.bert_utils import BertTraining
from experiment.bert_utils import InputExample, convert_examples_to_features


class BertTrainingPointwise(BertTraining):
    def __init__(self, *args, **kwargs):
        """This does random sampling of negative answers. Optionally, if n_train_answers is set to a value greater than
        one, this will sample for each question *n_train_answers* negative answers, which are ranked with the currently
        trained model. The best-ranked answer is chosen as a negative sample.

        :param args:
        :param kwargs:
        """
        super(BertTrainingPointwise, self).__init__(*args, **kwargs)
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
        # loss_fct = MSELoss()
        # step_loss = loss_fct(model.bert(input_ids, segment_ids, input_mask), label_ids.view(-1))

        return step_loss

    def prepare_training(self, model, data):
        super(BertTrainingPointwise, self).prepare_training(model, data)

        # create features for all training examples
        for i, dataset in enumerate(data.datas_train):
            self.logger.debug('prepare training data for dataset {}'.format(dataset.archive.name))
            dataset_examples = []
            for pool in dataset.archive.train.qa:
                for gt in pool.ground_truth:
                    self.train_questions[i].append(pool.question)
                    dataset_examples.append(
                        InputExample(pool.question.metadata['id'], pool.question.text, gt.text, label=1.0)
                    )
            self.logger.debug('convert_examples_to_features for dataset {}'.format(dataset.archive.name))
            self.train_positive_instances[i] = convert_examples_to_features(
                dataset_examples, self.length_question + self.length_answer, model.tokenizer, self.logger
            )
            self.logger.debug('done for dataset {}'.format(dataset.archive.name))

        # calculate number of batches
        min_examples = min([len(v) for v in self.train_questions.values()])
        n_datasets = len(self.train_questions)
        if self.epoch_max_examples is None: #or min_examples * 2 * n_datasets < self.epoch_max_examples:
            examples_per_epoch = min_examples * 2 * n_datasets
        else:
            examples_per_epoch = self.epoch_max_examples
        self.n_batches = math.ceil(examples_per_epoch / float(self.batchsize))

    def get_n_batches(self):
        return self.n_batches

    def get_next_batch(self, model, data):
        self.cur_dataset = (self.cur_dataset + 1) % len(self.train_questions)
        batchsize_half = int(self.batchsize / 2)
        indices = self.get_batch_indices(self.cur_dataset, batchsize_half)

        batch_examples = []
        prediction_examples = []

        unrelated_answers_all_qs = self.get_random_answers(self.cur_dataset, self.n_train_answers * len(indices))
        for idx, example_i in enumerate(indices):
            batch_examples.append(self.train_positive_instances[self.cur_dataset][example_i])

            unrelated_answers = unrelated_answers_all_qs[idx * self.n_train_answers:(idx + 1) * self.n_train_answers]
            prediction_examples += [
                InputExample(self.train_questions[self.cur_dataset][example_i].metadata['id'],
                             self.train_questions[self.cur_dataset][example_i].text,
                             a.text, label=0.0)
                for a in unrelated_answers
            ]

        prediction_examples = convert_examples_to_features(
            prediction_examples, self.length_question + self.length_answer, model.tokenizer, self.logger,
            show_example=False
        )

        # we only execute all this negative sampling using the network IF we choose between more than one neg. answer
        if self.n_train_answers > 1:
            prediction_results = []
            model.bert.eval()
            for predict_batch in range(int(math.ceil(len(prediction_examples) / float(self.batchsize_neg_ranking)))):
                batch_start_idx = predict_batch * self.batchsize_neg_ranking
                predict_batch_examples = prediction_examples[
                                         batch_start_idx: batch_start_idx + self.batchsize_neg_ranking]

                input_ids = torch.tensor([f.input_ids for f in predict_batch_examples], dtype=torch.long).to(
                    model.device)
                input_mask = torch.tensor([f.input_mask for f in predict_batch_examples], dtype=torch.long).to(
                    model.device)
                segment_ids = torch.tensor([f.segment_ids for f in predict_batch_examples], dtype=torch.long).to(
                    model.device)
                with torch.no_grad():
                    scores = model.bert(input_ids, segment_ids, input_mask)
                    prediction_results += scores.squeeze(dim=1).tolist()

            for count, example_i in enumerate(indices):
                predictions = prediction_results[self.n_train_answers * count:self.n_train_answers * (count + 1)]
                incorrect_example = prediction_examples[np.argmax(predictions) + self.n_train_answers * count]
                batch_examples.append(incorrect_example)
        else:
            batch_examples += prediction_examples

        self.batch_i += 1
        return batch_examples


component = BertTrainingPointwise
