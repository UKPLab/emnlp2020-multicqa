import os

import numpy as np
from unidecode import unidecode

from experiment.qa.data import QAData
from experiment.qa.data.models import *
from experiment.qa.data.reader import TSVArchiveReader
from experiment.util import unique_items


class AskUbuntuData(QAData):
    def _get_train_readers(self):
        return [AskUbuntuReader(self.config['askubuntu'], self.lowercased, self.logger)]


class AskUbuntuReader(TSVArchiveReader):
    def file_path(self, filename):
        return os.path.join(self.archive_path, filename)

    def read_items(self):
        items_path = self.file_path('text_tokenized.txt.gz')
        items = dict()
        for line in self.read_tsv(items_path, is_gzip=True):
            id = line[0]
            if len(line) < 2:
                self.logger.info('Item without title! Adding empty string for id {}'.format(id))
                text = ' '
            else:
                text = unidecode(line[1])
            # TODO make title-only / title+body configurable
            if len(line) > 2:
                text += ' ' + unidecode(line[2])

            body = unidecode(line[2]) if len(line) > 2 else ' '

            ti = TextItem(text)
            ti.metadata['id'] = id
            ti.metadata['body'] = body
            items[id] = ti
        return items

    def read_split(self, name, questions):
        split_path = self.file_path('{}.txt'.format(name))
        datapoints = []
        split_questions = []
        for i, line in enumerate(self.read_tsv(split_path, is_gzip=False)):
            query_question = questions[line[0]]
            ground_truth = [questions[gt_id] for gt_id in line[1].split()]
            pool = [questions[pa_id] for pa_id in line[2].split()]
            np.random.shuffle(pool)
            datapoints.append(QAPool(query_question, pool, ground_truth))

            split_questions += [query_question] + ground_truth + pool

        # we filter out all pools that do not contain any ground truth answer (except train!)
        if name != 'train_random':
            qa_pools_len_before = len(datapoints)
            datapoints = [p for p in datapoints if len([1 for gt in p.ground_truth if gt in p.pooled_answers]) > 0]
            qa_pools_len_after = len(datapoints)
            self.logger.info("Split {} reduced to {} item from {} due to missing ground truth in pool".format(
                name, qa_pools_len_after, qa_pools_len_before
            ))

        return Data('askubuntu / {}'.format(name), datapoints, unique_items(split_questions))

    def read(self):
        items = self.read_items()

        train = self.read_split("train_random", items)
        valid = self.read_split("dev", items)
        test = self.read_split("test", items)

        return Archive(self.name, train, valid, [test], list(items.values()), list(items.values()))


component = AskUbuntuData
