import json
from collections import OrderedDict
from os import path

import numpy as np

from experiment.qa.data import QAData
from experiment.qa.data.models import TextItem, QAPool, Data, Archive
from experiment.qa.data.reader import TSVArchiveReader


def _get_text_item(text, id):
    ti = TextItem(text)
    ti.metadata['id'] = id
    return ti


class WikiPassageQAReader(TSVArchiveReader):
    def read_split(self, name, answers):
        datapoints = []
        split_answers = []

        with open(path.join(self.archive_path, '{}.tsv'.format(name)), 'r') as f:
            next(f)
            for l in f:
                qid, question, doc_id, _, relevant_passages = l.strip().split('\t')

                question_ti = TextItem(question.lower() if self.lowercased else question)
                question_ti.metadata['id'] = 'question-{}'.format(qid)

                pool = [a for (k, a) in answers.items() if k.startswith('{}_'.format(doc_id))]
                np.random.shuffle(pool)
                ground_truth = [answers[doc_id + '_' + a_id] for a_id in relevant_passages.split(',')]

                datapoints.append(QAPool(question_ti, pool, ground_truth))
                split_answers += pool

        return Data('wikipassageqa / {}'.format(name), datapoints, split_answers)

    def read(self):
        answers = OrderedDict()
        with open(path.join(self.archive_path, 'document_passages.json'), 'r') as f:
            for document_id, passages in json.loads(f.read()).items():
                for passage_id, passage_text in passages.items():
                    answer_ti = TextItem(passage_text.lower() if self.lowercased else passage_text)
                    answer_ti.metadata['id'] = 'answer-{}-{}'.format(document_id, passage_id)
                    answers['{}_{}'.format(document_id, passage_id)] = answer_ti

        train = self.read_split('train', answers)
        valid = self.read_split('dev', answers)
        test = self.read_split('test', answers)

        questions = [qa.question for qa in (train.qa + valid.qa + test.qa)]
        answers = train.answers + valid.answers + test.answers

        return Archive(self.name, train, valid, [test], questions, answers)


class WikiPassageQAData(QAData):
    def _get_train_readers(self):
        return [WikiPassageQAReader(self.config['wikipassageqa'], self.lowercased, self.logger)]


component = WikiPassageQAData
