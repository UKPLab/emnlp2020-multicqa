from __future__ import division

import torch
import numpy as np

from experiment.bert_utils import InputExample, convert_examples_to_features
from experiment.qa.evaluation import BasicQAEvaluation


class QAEvaluationPointwise(BasicQAEvaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationPointwise, self).__init__(config, config_global, logger)
        self.length_question = self.config_global['question_length']
        self.length_answer = self.config_global['answer_length']

        self._cache = dict()

    def start(self, model, data, valid_only=False):
        model.bert.eval()
        return super(QAEvaluationPointwise, self).start(model, data, valid_only)

    def score(self, qa_pairs, model, data, task=None):
        input_ids_lst = []
        input_mask_lst = []
        segment_ids_lst = []
        label_ids_lst = []

        for (q, a, label) in qa_pairs:
            cache_id = '{}-{}'.format(id(q), id(a))
            example_features = self._cache.get(cache_id)

            if example_features is None:
                example = InputExample(q.metadata['id'], q.text, a.text, label=label)
                example_features = convert_examples_to_features([example], self.length_question + self.length_answer,
                                             model.tokenizer, self.logger)[0]
                self._cache[cache_id] = example_features

            input_ids_lst.append(example_features.input_ids)
            input_mask_lst.append(example_features.input_mask)
            segment_ids_lst.append(example_features.segment_ids)
            label_ids_lst.append(example_features.label_id)

        input_ids = torch.tensor(input_ids_lst, dtype=torch.long).to(model.device)
        input_mask = torch.tensor(input_mask_lst, dtype=torch.long).to(model.device)
        segment_ids = torch.tensor(segment_ids_lst, dtype=torch.long).to(model.device)
        label_ids = torch.tensor(label_ids_lst, dtype=torch.float).to(model.device)

        with torch.no_grad():
            scores = model.bert(input_ids, segment_ids, input_mask, label_ids, tasks=task)
            # loss_fct = nn.BCELoss()
            # loss = loss_fct(scores, label_ids.view(-1, 1))

        return scores.squeeze(dim=1).cpu().numpy(), None  # loss.cpu().numpy()


component = QAEvaluationPointwise
