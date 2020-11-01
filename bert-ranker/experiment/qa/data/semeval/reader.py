import os
from collections import defaultdict

import numpy as np
from bs4 import BeautifulSoup

from experiment.qa.data import QAData
from experiment.qa.data.models import *
from experiment.qa.data.reader import ArchiveReader
from experiment.util import unique_items


class SemEvalData(QAData):
    def _get_train_readers(self):
        return [SemEvalReader(self.config['path'], self.lowercased, self.logger)]


class SemEvalReader(ArchiveReader):
    def file_path(self, filename):
        return os.path.join(self.archive_path, filename)

    def read_xml(self, xml_path):
        questions = dict()
        pools = defaultdict(lambda: list())

        with open(self.file_path(xml_path), "r") as f:
            soup = BeautifulSoup(f.read(), 'lxml')
        for q in soup.find_all("orgquestion"):
            q_id = q.attrs["orgq_id"]
            q_title = q.find("orgqsubject").text
            q_body = q.find("orgqbody").text
            if q_id not in questions:
                q_text = q_title + ' ' + q_body
                if self.lowercased:
                    q_text = q_text.lower()
                questions[q_id] = (q_text, None)

            relq = q.find('relquestion')
            relq_id = relq.attrs["relq_id"]
            relq_title = relq.find("relqsubject").text
            relq_body = relq.find("relqbody").text
            relq_relevance = relq.attrs["relq_relevance2orgq"]
            relq_search_rank = relq.attrs["relq_ranking_order"]
            assert relq_id not in questions
            relq_text = relq_title + ' ' + relq_body
            if self.lowercased:
                relq_text = relq_text.lower()
            questions[relq_id] = (relq_text, relq_search_rank)
            pools[q_id].append((relq_id, relq_relevance))

        return questions, pools

    def read_split(self, name, *files):
        all_questions = dict()
        all_pools = []

        for i, file in enumerate(files):
            id_prefix = '{}-file{}-'.format(name, i)
            questions, pools = self.read_xml(file)
            org_q_ids = []
            for q_id, (text, search_engine_rank) in questions.items():
                if '_' not in q_id:
                    org_q_ids.append(q_id)

                full_q_id = id_prefix + q_id
                ti = TextItem(text)
                ti.metadata['id'] = full_q_id
                ti.metadata['search_engine_rank'] = search_engine_rank
                all_questions[full_q_id] = ti

            for q_id in org_q_ids:
                pooled_qs = []
                ground_truth = []
                for rel_q_id, label in pools[q_id]:
                    full_rel_q_id = id_prefix + rel_q_id
                    pooled_qs.append(all_questions[full_rel_q_id])
                    if label.lower() in ['perfectmatch', 'relevant']:
                        ground_truth.append(all_questions[full_rel_q_id])

                org_q = all_questions[id_prefix + q_id]
                all_pools.append(QAPool(org_q, pooled_qs, ground_truth))

        return Data('SemEval3b / {}'.format(name), all_pools, list(all_questions.values()))

    def read(self):
        train = self.read_split("train",
                                "train/SemEval2016-Task3-CQA-QL-train-part1.xml",
                                "train/SemEval2016-Task3-CQA-QL-train-part2.xml")
        valid = self.read_split("dev",
                                "dev/SemEval2016-Task3-CQA-QL-dev.xml")
        test16 = self.read_split("test-16",
                                 "test/English/SemEval2016-Task3-CQA-QL-test.xml")

        all_items = train.questions + valid.questions
        return Archive(self.name, train, valid, [test16], all_items, all_items)


component = SemEvalData
