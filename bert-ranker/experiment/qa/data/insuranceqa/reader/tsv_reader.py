import os
import random
from collections import defaultdict

import numpy as np

from experiment.qa.data.models import *
from experiment.qa.data.reader import TSVArchiveReader


class TSVReader(TSVArchiveReader):
    def __init__(self, archive_path, lowercased, logger, create_random_pools):
        super(TSVReader, self).__init__(archive_path, lowercased, logger)
        self.create_random_pools = create_random_pools
        # if set to true, will generate random pools for the train split

    def file_path(self, filename):
        return os.path.join(self.archive_path, filename)

    def read_items(self, name, vocab):
        items_path = self.file_path('{}.tsv.gz'.format(name))
        items = dict()
        for line in self.read_tsv(items_path, is_gzip=True):
            id = line[0]
            text_ids = line[1] if len(line) > 1 else ''
            tokens_text = ' '.join([vocab[t] for t in text_ids.split()])
            answer = TextItem(tokens_text)
            answer.metadata['id'] = id
            items[id] = answer

        return items

    def read_split(self, name, questions, answers):
        split_path = self.file_path('{}.tsv.gz'.format(name))
        datapoints = []
        split_answers = []
        for i, line in enumerate(self.read_tsv(split_path, is_gzip=True)):
            question = questions[line[0]]
            ground_truth = [answers[gt_id] for gt_id in line[1].split()]
            pool = [answers[pa_id] for pa_id in line[2].split()] if len(line) > 2 else None

            if pool is not None:
                np.random.shuffle(pool)

            datapoints.append(QAPool(question, pool, ground_truth))

            if pool is not None:
                split_answers += pool
            else:
                split_answers += ground_truth

        # we filter out all pools that do not contain any ground truth answer
        if name != 'train':
            # we filter out all pools that do not contain any ground truth answer
            qa_pools_len_before = len(datapoints)
            datapoints = [p for p in datapoints if len([1 for gt in p.ground_truth if gt in p.pooled_answers]) > 0]
            qa_pools_len_after = len(datapoints)
            self.logger.info("Split {} reduced to {} item from {} due to missing ground truth in pool".format(
                name, qa_pools_len_after, qa_pools_len_before
            ))

        return Data('tsv({}) / {}'.format(os.path.basename(self.archive_path), name), datapoints, split_answers)

    def read(self):
        vocab_filename = 'vocab.tsv.gz'
        vocab = dict(self.read_tsv(self.file_path(vocab_filename), is_gzip=True))

        answers = self.read_items('answers', vocab)

        q_filename = 'questions'
        questions = self.read_items(q_filename, vocab)

        try:
            additional_answers = self.read_items("additional-answers", vocab)
            additional_answers = list(additional_answers.values())
            self.logger.info('Read {} additional (unrelated) answers'.format(len(additional_answers)))
        except:
            additional_answers = None
            self.logger.info('No additional answers found for this dataset')

        train = self.read_split("train", questions, answers)
        valid = self.read_split("valid", questions, answers)
        test = self.read_split("test", questions, answers)

        if self.create_random_pools:
            for qa in train.qa:
                qa.pooled_answers = random.sample(train.answers, 100)

        return Archive(
            self.name, train, valid, [test],
            list(questions.values()), list(answers.values()),
            additional_answers
        )
