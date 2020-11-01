import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from numpy.linalg import norm
from tqdm import trange

from experiment.qa.evaluation import BasicQAEvaluation


class QAEvaluationUSE(BasicQAEvaluation):
    def __init__(self, config, config_global, logger):
        super(QAEvaluationUSE, self).__init__(config, config_global, logger)
        self.module_url = config["use"]["model"]
        self.batch_size = config["use"].get("batchsize", -1)

    def start(self, model, data, valid_only=False):
        return super(QAEvaluationUSE, self).start(model, data, valid_only)

    def score(self, qa_pairs, model, data, tasks):
        query_examples = [q.text for (q, _, _) in qa_pairs]
        doc_examples = [a.text for (_, a, _) in qa_pairs]

        g = tf.Graph()
        with g.as_default():
            text_input = tf.placeholder(dtype=tf.string, shape=[None])
            self.embed = hub.Module(self.module_url)
            embedded_text = self.embed(text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        with tf.compat.v1.Session(graph=g) as session:
            session.run(init_op)
            scores = []
            last_idx = 0
            for i in trange(1, int(len(query_examples) / self.batch_size)):
                repr_queries = session.run(embedded_text,
                                           feed_dict={text_input: query_examples[
                                                                  (i - 1) * self.batch_size:i * self.batch_size]})
                repr_docs = session.run(embedded_text,
                                        feed_dict={
                                            text_input: doc_examples[(i - 1) * self.batch_size:i * self.batch_size]})
                last_idx = i * self.batch_size
                repr_queries = np.stack(repr_queries)
                repr_docs = np.stack(repr_docs)
                scores.append(
                    (repr_queries * repr_docs).sum(axis=1) / (norm(repr_queries, axis=1) * norm(repr_docs, axis=1)))
            repr_queries = session.run(embedded_text,
                                       feed_dict={text_input: query_examples[last_idx:]})
            repr_docs = session.run(embedded_text,
                                    feed_dict={text_input: doc_examples[last_idx:]})
            repr_queries = np.stack(repr_queries)
            repr_docs = np.stack(repr_docs)

            scores.append(
                (repr_queries * repr_docs).sum(axis=1) / (norm(repr_queries, axis=1) * norm(repr_docs, axis=1)))

        scores = np.concatenate(scores)
        return scores, np.zeros(1)


component = QAEvaluationUSE
