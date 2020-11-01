#
# This is the code from our EMNLP-19 Submission. We changed it to remove the tokenization, which now depends on the
# model, e.g., BERT
#

import gzip
import io
import logging
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from os import path

import click
import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from numpy import linalg as LA
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

POST_TYPE_QUESTION = '1'
POST_TYPE_ANSWER = '2'

random.seed(1234)


class SEDataReader(object):
    """
    NOTE: - a typical xml string for question in original data looks like
                    <row Id="4" PostTypeId="1" AcceptedAnswerId="7" CreationDate="2008-07-31T21:42:52.667" Score="543" ViewCount="34799" Body="&lt;p&gt;I want to use a track-bar to change a form's opacity.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;This is my code:&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;decimal trans = trackBar1.Value / 5000;&#xA;this.Opacity = trans;&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;When I build the application, it gives the following error:&lt;/p&gt;&#xA;&#xA;&lt;blockquote&gt;&#xA;  &lt;p&gt;Cannot implicitly convert type &lt;code&gt;'decimal'&lt;/code&gt; to &lt;code&gt;'double'&lt;/code&gt;.&lt;/p&gt;&#xA;&lt;/blockquote&gt;&#xA;&#xA;&lt;p&gt;I tried using &lt;code&gt;trans&lt;/code&gt; and &lt;code&gt;double&lt;/code&gt; but then the control doesn't work. This code worked fine in a past VB.NET project.&lt;/p&gt;&#xA;" OwnerUserId="8" LastEditorUserId="3151675" LastEditorDisplayName="Rich B" LastEditDate="2017-09-27T05:52:59.927" LastActivityDate="2018-02-22T16:40:13.577" Title="While applying opacity to a form, should we use a decimal or a double value?" Tags="&lt;c#&gt;&lt;winforms&gt;&lt;type-conversion&gt;&lt;decimal&gt;&lt;opacity&gt;" AnswerCount="13" CommentCount="1" FavoriteCount="39" CommunityOwnedDate="2012-10-31T16:42:47.213" />

    """

    num_skipped_sample_none = 0
    num_skipped_other_posttype = 0
    num_retrieved = 0

    def __init__(self, data_file_path):
        """
        data_file_path : (string) path to the posts.xml file
        """
        self.data_file_path = data_file_path

    def question_data_filter(self, sample):
        return dict((k, sample.get(k, None)) for k in
                    ['Id', 'Title', 'Body', 'Tags', 'ParentId', 'PostTypeId', 'AcceptedAnswerId', 'Score'])

    def n_items_unfiltered(self):
        out = subprocess.check_output(['wc', '-l', self.data_file_path])
        return int(out.split()[0])

    def read_items(self, allowed_post_types=(POST_TYPE_QUESTION,), min_score=0, max_year=None):
        with io.open(self.data_file_path, 'r', encoding="utf-8") as f:
            for l in tqdm(f):
                try:
                    sample = ET.fromstring(l.strip()).attrib
                except ET.ParseError as e:
                    logger.info('(Ignoring) ERROR in parsing line (QUESTION READER):\n{}\n'.format(l.strip()))
                    sample = None
                if sample:
                    has_min_score = int(sample['Score']) >= min_score
                    has_max_year = True if max_year is None else int(sample['CreationDate'][:4]) <= max_year

                    if sample['PostTypeId'] in allowed_post_types and has_max_year and has_min_score:
                        SEDataReader.num_retrieved += 1
                        filtered_sample = self.question_data_filter(sample)
                        yield filtered_sample
                    else:
                        SEDataReader.num_skipped_other_posttype += 1

                else:
                    SEDataReader.num_skipped_sample_none += 1


def _clean_text(t):
    soup = BeautifulSoup(t, 'lxml')
    res = ' '.join([''.join(i.splitlines()) for i in soup.strings])
    res = res.replace(u"\\'", u" '")
    res = res.replace(u'\t', u'')
    return res


def read_questions(path):
    logger.info('Reading questions')
    reader = SEDataReader(path + "/Posts.xml")
    r = reader.read_items(POST_TYPE_QUESTION)

    questiondict = {}
    questionids = []

    en_stopwords = set(stopwords.words('english'))

    skipped = 0
    for item in r:
        question = item.get('Id')
        title = item.get('Title')
        body = item.get('Body')

        title_toks = title.lower().split()
        body_toks = body.lower().split()
        # We ensure that title and body are not the same, and are non-empty
        if len(title_toks) > 3 and len(body_toks) > 3 and title != body and len(
                en_stopwords & set(title_toks)) > 0 and len(en_stopwords & set(body_toks)) > 0:

            questionids.append(question)
            q = {'TITLE': title, 'BODY': body}
            questiondict[question] = q
        else:
            skipped += 1
            logger.debug('skipping question')

    logger.info('Skipped questions: {}'.format(skipped))

    return questiondict, questionids


@click.command()
@click.option('--se_path', required=True,
              help='Path pointing to the folder of the StackExchange dump (that contains the Posts.xml file)')
@click.option('--excluded_ids_path',
              help='Path of the file that contains the excluded ids (line by line)')
@click.option('--target_folder', required=True, help='Output folder (will be created)')
@click.option('--n_train_queries', required=True, type=int, help='Number of training queries')
@click.option('--n_dev_queries', required=True, type=int, help='Number of development queries')
@click.option('--n_dev_queries_max_percentage', required=True, default=0.2, type=float, help='Max % of dev/train')
@click.option('--n_max_questions', required=True, type=int,
              help='Maximum number of total questions (for negative sampling during training)')
@click.option('--pool_size', required=True, type=int,
              help='Number of negative examples that are chosen for the dev split')
@click.option('--pooling', type=click.Choice(['none', 'use', 'p-means']), help='Use sentence embeddings and FAISS to '
                                                                               'create candidate pools')
@click.option('--gpu', is_flag=True, help='Use the GPU Index for FAISS')
def create_data(se_path, excluded_ids_path, target_folder, n_train_queries, n_dev_queries, n_dev_queries_max_percentage,
                n_max_questions, pool_size, pooling, gpu=False):
    questiondict, qids = read_questions(se_path)

    if not path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    if excluded_ids_path:
        with open(excluded_ids_path, 'r') as f:
            excluded_ids = set([l.strip() for l in f])
    else:
        excluded_ids = set()

    logger.info('len(excluded_ids)={}'.format(len(excluded_ids)))
    qids_other = [qid for qid in qids if qid not in excluded_ids]
    logger.info('len(qids_other)={}'.format(len(qids_other)))

    sampled_qids = qids_other if len(qids_other) <= n_max_questions else random.sample(qids_other, n_max_questions)
    # the number of included questions in the dataset. this can be more than the number of queries so that we have
    # more variation for random sampling while not including too many queries (for large forums)

    logger.info('N sampled_qids = {}'.format(len(sampled_qids)))

    # adjust number of dev queries, if needed
    n_dev_queries = min(int(len(sampled_qids) * n_dev_queries_max_percentage), n_dev_queries)
    logger.info('n_dev_queries_max_percentage={}'.format(n_dev_queries_max_percentage))
    logger.info('n_questions={}'.format(len(qids)))
    logger.info('n_dev_queries={}'.format(n_dev_queries))

    n_sample_train_dev = n_train_queries + n_dev_queries
    if len(qids_other) < n_sample_train_dev:
        logger.info('Number of questions in SE dump less than train+dev (={})'.format(n_sample_train_dev))
        n_sample_train_dev = len(qids_other)

    sampled_qids_train_dev = random.sample(sampled_qids, n_sample_train_dev)
    sampled_qids_train = sampled_qids_train_dev[:-n_dev_queries]
    sampled_qids_dev = sampled_qids_train_dev[-n_dev_queries:]

    logger.info('N sampled_qids_train = {}'.format(len(sampled_qids_train)))
    logger.info('N sampled_qids_dev = {}'.format(len(sampled_qids_dev)))

    with gzip.open(target_folder + "/questions.tsv.gz", 'wt', encoding='utf-8') as f:
        for qid in sampled_qids:
            title = _clean_text(questiondict[qid]['TITLE'])
            body = _clean_text(questiondict[qid]['BODY'])
            f.write('{}\t{}\t{}\n'.format(qid, title, body))

    with gzip.open(target_folder + "/train.tsv.gz", 'wt', encoding='utf-8') as f:
        for qid in sampled_qids_train:
            f.write('{}\n'.format(qid))

    with gzip.open(target_folder + "/dev.tsv.gz", 'wt', encoding='utf-8') as f:
        if pooling != "none":
            # pooling with USE sentence embeddings over question titles and similarity search
            logger.info('Building FAISS index with sentence embeddings of Q titles')
            g = tf.Graph()
            with g.as_default():
                sentences_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
                if pooling == 'use':
                    # sentence_encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
                    sentence_encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-qa/3")
                    lowercase = False
                    dim = 512
                elif pooling == 'p-means':
                    sentence_encoder = hub.Module("https://public.ukp.informatik.tu-darmstadt.de/arxiv2018-xling-sentence-embeddings/tf-hub/monolingual/1")
                    lowercase = True
                    dim = 3600
                else:
                    raise Exception('Unknown pooling method "{}"'.format(pooling))
                graph = sentence_encoder(sentences_placeholder)
                init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
            g.finalize()

            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            with tf.Session(graph=g, config=sess_config) as sess:
                sess.run(init_op)
                titles = [questiondict[qid]['TITLE'] for qid in sampled_qids]
                if lowercase:
                    titles = [t.lower() for t in titles]
                logger.info('Computing all embeddings...')
                embeddings = np.empty((0, dim)).astype('float32')

                for i in tqdm(range(0, len(sampled_qids), 128)):
                    e = sess.run(graph, feed_dict={sentences_placeholder: titles[i:i + 128]})
                    embeddings = np.vstack((embeddings, e))
            tf.reset_default_graph()    # release memory

            logger.info('Normalizing embeddings...')
            embeddings = embeddings / LA.norm(embeddings, axis=0)
            # normalize embeddings so that IP-index does cosine similarity

            logger.info('Adding embeddings to FAISS index...')
            logger.info('embeddings shape: {}'.format(embeddings.shape))
            index = faiss.IndexFlatIP(512)
            index.add(embeddings)
            if gpu:
                res = faiss.StandardGpuResources()  # use a single GPU
                index = faiss.index_cpu_to_gpu(res, 0, index)

            logger.info('Querying FAISS...')
            for i, qid in tqdm(enumerate(sampled_qids_dev)):
                embedding = index.reconstruct(sampled_qids.index(qid))
                _, similar_items = index.search(np.reshape(embedding, [1, -1]), pool_size)
                similar_items_qids = [sampled_qids[j] for j in similar_items[0]]
                neg = ' '.join(similar_items_qids)
                f.write('{}\t{}\n'.format(qid, neg))

                # print some examples
                if i < 3:
                    logger.info('Query: {}'.format(questiondict[qid]['TITLE']))
                    for qid_similar in similar_items_qids[:3]:
                        logger.info('=>: {}'.format(questiondict[qid_similar]['TITLE']))
                    logger.info('-' * 10)
        else:
            # pooling with random sampling
            for qid in sampled_qids_dev:
                neg = ' '.join([str(i) for i in random.sample(sampled_qids_train_dev, pool_size)])
                f.write('{}\t{}\n'.format(qid, neg))


if __name__ == "__main__":
    create_data()
