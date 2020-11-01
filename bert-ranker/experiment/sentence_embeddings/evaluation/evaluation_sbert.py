import os

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from experiment.qa.evaluation import BasicQAEvaluation


class QAEvaluationSBert(BasicQAEvaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationSBert, self).__init__(config, config_global, logger)
        os.environ['TORCH_HOME'] = config["sbert"]["cache_dir"]
        self.model = SentenceTransformer(config["sbert"]["model"])
        self.batch_size = config["batchsize"]

    def start(self, model, data, valid_only=False):
        return super(QAEvaluationSBert, self).start(model, data, valid_only)

    def score(self, qa_pairs, model, data, tasks):
        query_examples = [q.text for (q, _, _) in qa_pairs]
        doc_examples = [a.text for (_, a, _) in qa_pairs]

        repr_queries = torch.from_numpy(np.stack(self.model.encode(query_examples, batch_size=self.batch_size, show_progress_bar=False)))
        repr_docs = torch.from_numpy(np.stack(self.model.encode(doc_examples, batch_size=self.batch_size, show_progress_bar=False)))
        scores = torch.cosine_similarity(repr_queries, repr_docs)

        return scores.cpu().numpy(), np.zeros(1)


component = QAEvaluationSBert
