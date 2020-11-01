import os
from os import path
import random

import numpy as np
from unidecode import unidecode

from experiment.qa.data import QAData
from experiment.qa.data.models import TextItem, QAPool, Data, Archive
from experiment.qa.data.reader import TSVArchiveReader
from experiment.util import unique_items


class SEQuestionsData(QAData):
    def _get_train_readers(self):
        return [SEQuestionsReader(self.config['path'], self.lowercased, self.logger)]


class SEQuestionsReader(TSVArchiveReader):
    def __init__(self, archive_path, lowercased, logger):
        super().__init__(archive_path, lowercased, logger)

    def file_path(self, filename):
        return os.path.join(self.archive_path, filename)

    def read_items(self):
        questions_path = self.file_path('questions.tsv.gz')
        items = dict()
        for line in self.read_tsv(questions_path, is_gzip=True):
            id = line[0]
            title = unidecode(line[1])
            if len(line) > 2:
                body = unidecode(line[2])
            else:
                self.logger.info('Question has no body! {}'.format(id))
                body = title

            answer = unidecode(line[3]).strip() if len(line) > 3 else None
            duplicates = line[4].split(',') if len(line) > 4 else []

            if self.lowercased:
                title = title.lower()
                body = body.lower()
                answer = answer.lower() if answer else None

            ti = TextItem(title)
            ti.metadata['id'] = '{}_title'.format(id)
            ti.metadata['duplicates'] = duplicates
            items[ti.metadata['id']] = ti
            ti = TextItem(body)
            ti.metadata['id'] = '{}_body'.format(id)
            items[ti.metadata['id']] = ti
            if answer:
                ti = TextItem(answer)
                ti.metadata['id'] = '{}_answer'.format(id)
                items[ti.metadata['id']] = ti

        return items

    def read_split(self, name, questions):
        file_path = self.file_path('{}.tsv.gz'.format(name))
        question_keys = [k for k in questions.keys() if k.endswith('body') or k.endswith('answer')]

        datapoints = []
        split_questions = []
        for line in self.read_tsv(file_path, is_gzip=True):
            question_id = line[0]
            query = questions[question_id + '_title']
            truth = [questions[question_id + '_body']]
            if name == 'train':
                # we also include answers and bodies of duplicates, if they are in the dataset
                answer = questions.get(question_id + '_answer')
                if answer:
                    truth.append(answer)
                for dup_id in query.metadata['duplicates']:
                    other_gt = [
                        questions.get(dup_id + '_body'),
                        questions.get(dup_id + '_answer')
                    ]
                    for ti in other_gt:
                        if ti:
                            truth.append(ti)

            pool = []
            if len(line) > 1:
                pool = [questions[neg_id + '_body'] for neg_id in line[1].split()]
                #np.random.shuffle(pool)
            else:
                # create a pool with random answers plus gt
                pool = (truth + [questions[neg_id] for neg_id in random.sample(question_keys, 20)])[:20]

            split_questions += [query] + truth + pool
            datapoints.append(QAPool(query, pool, truth))

        # We filter out all pools that do not contain any ground truth answer (i.e., body)
        # This can e.g. happen if the dataset is not in English but the retrieval model was in English only
        if name != 'train':
            # we filter out all pools that do not contain any ground truth answer
            qa_pools_len_before = len(datapoints)
            datapoints = [p for p in datapoints if len([1 for gt in p.ground_truth if gt in p.pooled_answers]) > 0]
            qa_pools_len_after = len(datapoints)
            self.logger.info("Split {} reduced to {} item from {} due to missing ground truth in pool".format(
                name, qa_pools_len_after, qa_pools_len_before
            ))

        return Data('SE({}) / {}'.format(os.path.basename(self.archive_path), name), datapoints,
                    unique_items(split_questions))

    def read(self):
        items = self.read_items()

        train = self.read_split("train", items)
        valid = self.read_split("dev", items)

        test = []
        if path.exists(self.file_path('test.tsv.gz')):
            test = [self.read_split("test", items)]

        return Archive(self.name, train, valid, test, list(items.values()), list(items.values()))


component = SEQuestionsData
