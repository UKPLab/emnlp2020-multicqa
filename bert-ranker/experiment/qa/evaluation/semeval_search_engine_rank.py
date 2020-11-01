from __future__ import division

import torch
import numpy as np

from experiment.bert_utils import InputExample, convert_examples_to_features
from experiment.qa.evaluation import BasicQAEvaluation


class QAEvaluationSemEvalSearchEngineRank(BasicQAEvaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationSemEvalSearchEngineRank, self).__init__(config, config_global, logger)

    def score(self, qa_pairs, model, data, task=None):
        scores = [-int(a.metadata['search_engine_rank']) for (_, a, _) in qa_pairs]
        return np.array(scores), np.zeros(1)


component = QAEvaluationSemEvalSearchEngineRank
