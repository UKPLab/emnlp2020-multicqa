from __future__ import division

import itertools
from collections import defaultdict, OrderedDict

import torch
from math import ceil
import os
import numpy as np
from numpy.linalg import norm
import pickle
from progressbar import progressbar
import experiment
from experiment.util import flatten
from experiment.bert_utils import InputExample, convert_examples_to_features


class BasicQAEvaluation(experiment.Evaluation):
    def __init__(self, config, config_global, logger):
        super(BasicQAEvaluation, self).__init__(config, config_global, logger)
        self.primary_measure = self.config.get('primary_measure', 'accuracy')
        self.batchsize = self.config.get('batchsize', 512)
        self.hinge_margin = self.config.get('hinge_margin', None)
        self.use_mean_adapter_selection = self.config.get("mean_adapter_selection", False)
        self.mean_adapter_similarity = self.config.get("mean_adapter_similarity", "cosine")
        self.log_outputs = self.config.get("log_outputs", False)

    def start(self, model, data, valid_only=False):
        """Will compute the results for all splits and datasets individually (and log them) but at the end average over
        the individual dataset results

        :param model:
        :param data:
        :param valid_only:
        :return:
        """
        if self.use_mean_adapter_selection:
            with open(os.path.join(self.config_global["output_path"], "adapter_mean_vectors.pickle"), "rb") as f:
                task_mean_vector_dict = pickle.load(f)
                self.tasks, self.vectors = zip(*[[task, task_mean_vector_dict[task]] for task in task_mean_vector_dict])
                self.vectors = np.concatenate(self.vectors, axis=0)
                self.vectors = self.vectors/norm(self.vectors, axis=1)[:,None]

        evaluation_data = [('dev', [(d.archive.valid, d.config) for d in data.datas_dev])]
        if not valid_only:
            evaluation_data += [('test', [[(t,d.config) for t in d.archive.test] for d in data.datas_test])]

        results_primary = dict()  # dict[string, float] (split name, value)
        results_all = defaultdict(lambda: defaultdict(lambda: list()))  # (split name, measure, values)
        for split_name, datasets in evaluation_data:
            datasets = flatten(datasets) if isinstance(datasets[0], list) else datasets
            for dataset, config in datasets:
                self.logger.info("Evaluating {}".format(dataset.split_name))
                ranks = []
                average_precisions = []
                p5 = []
                hinge_losses = []
                # we calculate hinge loss only if the config contains a value for the hinge_margin parameter (in eval)
                # losses = []
                scores = []

                if not self.use_mean_adapter_selection:
                    task = config.get("adapter")

                    # perform the scoring, used to calculate the measure values
                    qa_pairs = [(qa.question, a, 1.0 if a in qa.ground_truth else 0.0) for qa in dataset.qa for a in
                                qa.pooled_answers]
                    n_batches = int(ceil(len(qa_pairs) / float(self.batchsize)))

                    bar = self.create_progress_bar()
                    for i in bar(range(n_batches)):
                        batch_qa_pairs = qa_pairs[i * self.batchsize:(i + 1) * self.batchsize]
                        result = self.score(batch_qa_pairs, model, data, task)
                        scores += result[0].tolist()
                        # losses += [result[1].tolist()] * len(batch_qa_pairs)

                else:
                    chosen_tasks = {}
                    for pool in progressbar(dataset.qa, prefix="Pool "):
                        task = self._get_mean_adapter(model, pool)
                        if task in chosen_tasks:
                            chosen_tasks[task] += 1
                        else:
                            chosen_tasks[task] = 1
                        # self.logger.debug("Pool adapter chosen: {}".format(task))
                        task = [task]
                        qa_pairs = [(pool.question, a, 1.0 if a in pool.ground_truth else 0.0) for a in
                                    pool.pooled_answers]
                        n_batches = int(ceil(len(qa_pairs) / float(self.batchsize)))

                        for i in range(n_batches):
                            batch_qa_pairs = qa_pairs[i * self.batchsize:(i + 1) * self.batchsize]
                            result = self.score(batch_qa_pairs, model, data, task)
                            scores += result[0].tolist()
                            # losses += [result[1].tolist()] * len(batch_qa_pairs)

                scores_used = 0
                for pool in dataset.qa:
                    if self.log_outputs:
                        self.logger.debug("-" * 10)
                        self.logger.debug("Question\t\t\t\t\t{}".format(pool.question.text))

                    scores_pool = scores[scores_used:scores_used + len(pool.pooled_answers)]
                    scores_used += len(pool.pooled_answers)

                    sorted_answers = sorted(zip(scores_pool, pool.pooled_answers), key=lambda x: -x[0])

                    rank = 0
                    precisions = []
                    p5_components = []
                    pool_gold_labels = []
                    for i, (score, answer) in enumerate(sorted_answers, start=1):
                        if answer in pool.ground_truth:
                            if self.log_outputs:
                                self.logger.debug("Rank: {}, correct!, score={}\t\t{}".format(i, score, answer.text))
                            pool_gold_labels.append(1.0)
                            if rank == 0:
                                rank = i
                            precisions.append((len(precisions) + 1) / float(i))
                        else:
                            pool_gold_labels.append(0.0)
                            if i <= 5 and self.log_outputs:
                                self.logger.debug("Rank: {}, wrong!, score={}\t\t{}".format(i, score, answer.text))
                        if i <= 5:
                            p5_components.append(1.0 if answer in pool.ground_truth else 0.0)
                    p5.append(np.mean(p5_components))

                    if rank == 0:
                        self.logger.warn("RANK IS ZERO")
                        precisions.append(0)

                    ranks.append(rank)
                    average_precisions.append(np.mean(precisions))

                    if self.hinge_margin:
                        # we pair all positive answers with all negative answers to calculate hinge loss on dev/test
                        scores_pos = [s for (s, a) in zip(scores_pool, pool.pooled_answers) if a in pool.ground_truth]
                        scores_neg = [s for (s, a) in zip(scores_pool, pool.pooled_answers) if a not in pool.ground_truth]
                        hinge_losses += [n + self.hinge_margin - p for (p,n) in itertools.product(scores_pos, scores_neg)]

                    if not valid_only:
                        self.logger.debug('Rank: {}'.format(rank))

                correct_answers = len([a for a in ranks if a == 1])
                dataset_split_results = {
                    # 'loss': np.mean(losses),
                    'accuracy': correct_answers / float(len(ranks)),
                    'mrr': np.mean([1 / float(r) if r > 0 else 0 for r in ranks]),
                    'map': np.mean(average_precisions),
                    'p@5': np.mean(p5),
                    # 'hinge_loss': np.mean(hinge_losses),
                }

                for key, value in dataset_split_results.items():
                    results_all[split_name][key].append(value)

                self.logger.info('Results for dataset {}'.format(dataset.split_name))
                if self.use_mean_adapter_selection:
                    chosen_tasks = OrderedDict(sorted(chosen_tasks.items(), key=lambda item: -item[1]))
                    self.logger.info("Selected adapters: {}".format(chosen_tasks))
                self.logger.info('Correct answers: {}/{}'.format(correct_answers, len(dataset.qa)))
                self._log_results(dataset_split_results)

            for key, values in results_all[split_name].items():
                results_all[split_name][key] = np.mean(values)
            results_primary[split_name] = results_all[split_name][self.primary_measure]

            self.logger.info('Results for all datasets in {}'.format(split_name))
            self._log_results(results_all[split_name])

        return results_primary, results_all

    def prepare_data(self, model, data, valid_only=False):
        pass

    def score(self, question_answer_pairs, model, data, task=None):
        raise NotImplementedError()

    def _log_results(self, results):
        # self.logger.info('Loss: {}'.format(results['loss']))
        self.logger.info('Accuracy: {}'.format(results['accuracy']))
        self.logger.info('MRR: {}'.format(results['mrr']))
        self.logger.info('MAP: {}'.format(results['map']))
        self.logger.info('P@5: {}'.format(results['p@5']))

    def _get_mean_adapter(self, model, pool):
        question = [InputExample("", pool.question.text, label=1.0)]

        features = convert_examples_to_features(question, self.length_question + self.length_answer,
                                                model.tokenizer, self.logger)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(model.device)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(model.device)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(model.device)

        with torch.no_grad():
            q_vector = model.bert.average_standard_bert_output(input_ids, segment_ids, input_mask).cpu().numpy()
            q_vector = np.squeeze(q_vector/norm(q_vector))
        if self.mean_adapter_similarity == "cosine":
            similarity = 1 - np.dot(self.vectors, q_vector)
        task = self.tasks[np.argmin(similarity)]
        return task


component = BasicQAEvaluation
