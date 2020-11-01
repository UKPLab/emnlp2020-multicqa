import os
from collections import defaultdict

import numpy as np
from bs4 import BeautifulSoup

from experiment.qa.data import QAData
from experiment.qa.data.models import *
from experiment.qa.data.reader import ArchiveReader
from experiment.qa.data.semeval.reader import SemEvalReader
from experiment.util import unique_items


class SemEvalData17(QAData):
    def _get_train_readers(self):
        return [SemEvalReader17(self.config['path'], self.lowercased, self.logger)]


class SemEvalReader17(SemEvalReader):
    def read(self):
        train = self.read_split("train",
                                "sem-eval-2016-v3.2/train/SemEval2016-Task3-CQA-QL-train-part1.xml",
                                "sem-eval-2016-v3.2/train/SemEval2016-Task3-CQA-QL-train-part2.xml")
        valid = self.read_split("dev",
                                "sem-eval-2016-v3.2/dev/SemEval2016-Task3-CQA-QL-dev.xml")
        test17 = self.read_split("test-17",
                                 "test/English/SemEval2017-task3-English-test.xml")

        all_items = train.questions + valid.questions
        return Archive(self.name, train, valid, [test17], all_items, all_items)


component = SemEvalData17
