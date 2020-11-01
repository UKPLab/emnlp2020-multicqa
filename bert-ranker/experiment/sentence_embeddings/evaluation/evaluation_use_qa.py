import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from numpy.linalg import norm

from experiment.qa.evaluation import BasicQAEvaluation


class QAEvaluationUSEQA(BasicQAEvaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationUSEQA, self).__init__(config, config_global, logger)
        self.batch_size = config["batchsize"]

        self.module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-qa/3')
        self.i = 0

    def start(self, model, data, valid_only=False):
        return super(QAEvaluationUSEQA, self).start(model, data, valid_only)

    def get_reps(self, qs, docs):
        # encode the texts
        repr_queries = self.module.signatures['question_encoder'](tf.constant(qs))['outputs'].numpy()
        repr_docs = self.module.signatures['response_encoder'](
            input=tf.constant(docs),
            context=tf.constant(docs)
        )['outputs'].numpy()

        return repr_queries, repr_docs

    def score(self, qa_pairs, model, data, tasks):
        query_examples = [q.text[:4096] for (q, _, _) in qa_pairs]
        doc_examples = [a.text[:4096] for (_, a, _) in qa_pairs]
        q_reps, d_reps = self.get_reps(query_examples, doc_examples)

        scores = (q_reps * d_reps).sum(axis=1) / (norm(q_reps, axis=1) * norm(d_reps, axis=1))
        return scores, np.zeros(1)


component = QAEvaluationUSEQA
