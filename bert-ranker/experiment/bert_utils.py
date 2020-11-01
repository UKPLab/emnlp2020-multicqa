import os
import re
from collections import defaultdict
from glob import glob
from os import path

import numpy as np
import torch
# from pytorch_transformers import WarmupLinearSchedule
import transformers
from tensorboardX import SummaryWriter
from transformers import AdamW, RobertaTokenizer, BertTokenizer, AlbertTokenizer

from experiment import Model, Training
from experiment.checkpointing import QATrainStatePyTorch


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, max_seq_length, tokenizer, logger, show_example=True):
    """Loads a data file into a list of `InputBatch`s."""

    # tokens_a_longer_max_seq_length = 0
    features = []

    if isinstance(tokenizer, RobertaTokenizer):
        solo_special_tokens_num = 2
        pair_special_tokens_num = 4
        pair_b_special_tokens_num = 2
        add_solo_special_tokens = lambda tokens_a: [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
        add_pair_special_tokens = lambda t, tokens_b: t + [tokenizer.sep_token] + tokens_b + [tokenizer.sep_token]
    elif isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, AlbertTokenizer):
        solo_special_tokens_num = 2
        pair_special_tokens_num = 3
        pair_b_special_tokens_num = 1
        add_solo_special_tokens = lambda tokens_a: [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
        add_pair_special_tokens = lambda t, tokens_b: t + tokens_b + [tokenizer.sep_token]

    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        len_tokens_a = len(tokens_a)
        len_tokens_b = 0

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            len_tokens_b = len(tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - pair_special_tokens_num)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - solo_special_tokens_num:
                tokens_a = tokens_a[:(max_seq_length - solo_special_tokens_num)]

        # if (len_tokens_a + len_tokens_b) > (max_seq_length - solo_special_tokens_num):
        #    tokens_a_longer_max_seq_length += 1

        tokens = add_solo_special_tokens(tokens_a)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens = add_pair_special_tokens(tokens, tokens_b)
            segment_ids += [1] * (len(tokens_b) + pair_b_special_tokens_num)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += [tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]] * (max_seq_length - len(input_ids))
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        if show_example and ex_index < 1 and example.guid.startswith('train-'):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info(("tokens: %s" % " ".join(
                [str(x) for x in tokens])).encode("utf-8"))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    # logger.debug(":: Sentences longer than max_sequence_length: %d" % (tokens_a_longer_max_seq_length))
    # logger.debug(":: Num sentences: %d" % (len(examples)))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BertWrapperModel(Model):
    _MODEL_CLASS = None
    _TOKENIZER_CLASS = None
    _CONFIG_CLASS = None

    def __init__(self, config, config_global, logger):
        """This is general BERT wrapper model. It can be used for BERT CLS classification, and BERT repr. learning

        You must set _MODEL_CLASS, _TOKENIZER_CLASS and _CONFIG_CLASS to a superclass of one of Huggingface's model and tokenizer classes.
        :param config: The model configuration
        :param config_global: The global experiment configuation
        :param logger: The logger of the current experiment
        """
        super(BertWrapperModel, self).__init__(config, config_global, logger)
        self.question_length = self.config_global['question_length']
        self.answer_length = self.config_global['answer_length']
        self.bert_model_name = self.config['bert_model']
        self.lowercased = True if 'uncased' in self.bert_model_name else False

        self.bert = None
        self.tokenizer = None
        self.device = None

        self.create_adapters = self.config.get('create_adapters', [])
        self.adapter_downsample = self.config.get('adapter_downsample', None)
        self.adapter_attention_type = self.config.get("adapter_attention_type", None)

    def set_bert_configs_from_output_folder(self, epoch):
        checkpoints_path = path.join(self.config_global['output_path'], 'checkpoints')
        self.config["huggingface_config"] = path.join(checkpoints_path, "/huggingface-config.json")

        if epoch == 'best':
            checkpoints = sorted(glob(checkpoints_path + '/*.tar'), key=lambda x: int(re.findall(r'(\d+).tar', x)[0]))
            path_best_checkpoint = checkpoints[0]
        else:
            path_best_checkpoint = path.join(checkpoints_path, 'model-checkpoint-{}.tar'.format(epoch))

        self.config["base_checkpoint"] = path_best_checkpoint
        self.config["base_head_checkpoint"] = path_best_checkpoint

    def build(self, data):
        if "huggingface_config" in self.config and os.path.exists(self.config["huggingface_config"]):
            self.logger.info(
                "Creating model with config {}. Weights will not be initialised to a pretrained model. Use a checkpoint or 'model.base_checkpoint'".format(
                    self.config["huggingface_config"]))
            config = self._CONFIG_CLASS.from_json_file(self.config["huggingface_config"])
            self.bert = self._MODEL_CLASS(from_pretrained=False, config=config)
        else:
            self.bert = self._MODEL_CLASS(from_pretrained=True, model_name=self.bert_model_name,
                                          cache_dir=self.config.get('bert_cache_dir', None))

        if "base_checkpoint" in self.config and os.path.exists(self.config["base_checkpoint"]):
            self.logger.info("Loading base checkpoint from {}".format(self.config["base_checkpoint"]))
            checkpoint = torch.load(self.config["base_checkpoint"])["model"]
            bert_state_dict = self.bert.state_dict()
            # old checkpoint might be of model that inherits from BertModel, thus it has to many parameters which we remove
            delete_keys = []
            for key in checkpoint:
                if key not in bert_state_dict:
                    delete_keys.append(key)
            for key in delete_keys:
                checkpoint.pop(key)
            can_load_checkpoint = True
            missing_keys = []
            for key in bert_state_dict:
                if key not in checkpoint:
                    # not needed with load_state_dict(*, strict=True)
                    # can_load_checkpoint = False
                    missing_keys.append(key)
            if can_load_checkpoint:
                self.bert.load_state_dict(checkpoint, strict=False)
            else:
                self.logger.warn("Can not load the base checkpoint. Missing keys: {}".format(missing_keys))

        if "base_head_checkpoint" in self.config and os.path.exists(self.config["base_head_checkpoint"]):
            self.logger.info("Loading base head checkpoint from {}".format(self.config["base_head_checkpoint"]))
            checkpoint = torch.load(self.config["base_head_checkpoint"])["model"]
            head_state_dict = {}
            for key in checkpoint:
                if "bert" not in key:
                    head_state_dict[key] = checkpoint[key]
            self.logger.info("Loaded head parameters: {}".format(head_state_dict.keys()))
            self.bert.load_state_dict(head_state_dict, strict=False)

        # if self.adapter_attention_type is None:
        for adapter_name in self.create_adapters:
            if not hasattr(self.bert.bert.config, 'adapters') or adapter_name not in self.bert.bert.config.adapters:
                self.logger.info('Adding adapter "{}" with downsampling factor {} to BERT'.format(
                    adapter_name, self.adapter_downsample
                ))
                self.bert.bert.add_adapter(adapter_name, self.adapter_downsample)

        if self.adapter_attention_type is not None:
            if not hasattr(self.bert.bert.config, 'adapter_attention') \
                    or any([len(set(self.create_adapters) - set(adapters)) == 0
                            for adapters in self.bert.bert.config.adapter_attention]):
                self.logger.info('Adding adapter attention for adapters "{}" with type {} to BERT'.format(
                    self.create_adapters, self.adapter_attention_type))
                self.bert.bert.add_attention_layer(self.create_adapters, self.adapter_attention_type)

        self.tokenizer = self._TOKENIZER_CLASS.from_pretrained(self.bert_model_name,
                                                               cache_dir=self.config.get('bert_cache_dir', None),
                                                               do_lower_case=self.lowercased)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert.to(self.device)


class BertTraining(Training):
    def __init__(self, config, config_global, logger):
        super(BertTraining, self).__init__(config, config_global, logger)

        self.epoch_max_examples = config.get('epoch_max_examples')

        # checkpointing and weight restoring
        self.config_checkpointing = config.get('checkpointing', dict())
        self.state = QATrainStatePyTorch(self.checkpoint_folder,
                                         less_is_better=self.config_checkpointing.get('score_less_is_better', False),
                                         logger=self.logger)
        if self.config_checkpointing.get('remove_save_folder_on_start', False) is True \
                and os.path.isdir(self.checkpoint_folder):
            self.logger.info('Removing old save folder')
            self.state.clear()

        # tensorboard
        self.tensorboard = SummaryWriter(self.tensorboard_folder)

        self.n_train_answers = config['n_train_answers']
        self.length_question = self.config_global['question_length']
        self.length_answer = self.config_global['answer_length']
        self.n_epochs = self.config['epochs']
        self.backup_checkpoint_every = self.config.get('backup_checkpoint_every', -1)
        self.logger.debug('Checkpoint backup ' +
                          ('enabled ({} ep)'.format(
                              self.backup_checkpoint_every) if self.backup_checkpoint_every > 0 else 'disabled')
                          )

        self.chkpt_optimizer = self.config.get('checkpoint_optimizer', True)
        self.logger.debug('Checkpoint optimizer? {}'.format(self.chkpt_optimizer))

        self.batchsize = self.config['batchsize']
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.batchsize_neg_ranking = self.config.get('batchsize_neg_ranking', 64)
        if "train_adapter" in config:
            self.adapter_task = config["train_adapter"]
        else:
            self.adapter_task = None
        self.use_all_negative_sampling = self.config.get('use_all_negative_sampling', False)
        if self.use_all_negative_sampling:
            self.logger.warning('Using all answers for negative sampling!')

        # to enable multitasking capabilities we cache instances and questions separately for each dataset
        self.data = None
        self.train_positive_instances = defaultdict(lambda: list())
        self.train_questions = defaultdict(lambda: list())
        self.train_random_indices = defaultdict(lambda: list())
        self.train_negative_answers = defaultdict(lambda: list())

    def get_loss(self, model, step_examples, tasks=None):
        """Calls this method during training to obtain the loss of the model for some examples

        :param model: BertWrapperModel
        :param step_examples: list[InputExample] or list[(InputExample, InputExample, label)]
        :param tasks: list[str] or None of adapter name(s)
        :return:
        """
        raise NotImplementedError()

    def start(self, model, data, evaluation):
        """

        :param model:
        :type model: BertWrapperModel
        :param data:
        :type data: MultiData
        :type evaluation: Evaluation
        """
        self.prepare_training(model, data)

        if hasattr(model.bert.config, 'adapter_attention'):
            self.logger.info("Adapter attention detected. Freezing all weights except the adapter attention")
            for param in model.bert.bert.parameters():
                param.requires_grad = False
            model.bert.bert.enable_adapters(unfreeze_adapters=False, unfreeze_attention=True)
        elif hasattr(model.bert.config, 'adapters'):
            self.logger.info("Adapters detected. Freezing all weights except the adapters")
            for param in model.bert.bert.parameters():
                param.requires_grad = False
            model.bert.bert.enable_adapters(unfreeze_adapters=True, unfreeze_attention=False)

        if self.config.get('freeze_bert', False):
            self.logger.warn('FREEZING BERT')
            for name, param in model.bert.bert.named_parameters():
                self.logger.warn('freeze {}'.format(name))
                param.requires_grad = False

        if self.config.get('freeze_head', False):
            self.logger.info("Freezing the weights of the classification head")
            for name, param in model.bert.lin_layer.named_parameters():
                param.requires_grad = False

        # for param in model.bert.parameters():
        #     if param.requires_grad:
        #         print(param)

        # Prepare BERT optimizer
        param_optimizer = list(model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.get('weight_decay', 0.0)},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = self.get_n_batches() * self.n_epochs
        num_warmup_steps = int(self.get_n_batches() * self.config.get('warmup_proportion', 0.1))

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get('learning_rate', 5e-5),
            eps=self.config.get('adam_epsilon', 1e-8)
        )
        # correct_bias=False)

        if num_warmup_steps > 0:
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
        else:
            # scheduler = WarmupConstantSchedule(optimizer=optimizer, warmup_steps=num_warmup_steps)
            scheduler = transformers.get_constant_schedule(optimizer)

        self.state.load(model.bert, optimizer, weights='last')
        start_epoch = self.state.recorded_epochs + 1
        end_epoch = self.n_epochs + 1

        if self.state.recorded_epochs > 0:
            self.logger.info('Loaded the weights of last epoch {} with valid score={}'.format(
                self.state.recorded_epochs, self.state.scores[-1]
            ))
            # if start_epoch < end_epoch and not self.config.get('skip_restore_validation', False):
            #     self.logger.info('Now calculating validation score (to verify the restoring success)')
            #     valid_score = list(evaluation.start(model, data, valid_only=True)[0].values())[0]
            #     self.logger.info('Score={:.4f}'.format(valid_score))

        self.logger.info('Running from epoch {} to epoch {}'.format(start_epoch, end_epoch - 1))

        global_step = self.get_n_batches() * self.state.recorded_epochs

        for epoch in range(start_epoch, end_epoch):
            self.logger.info('Epoch {}/{}'.format(epoch, self.n_epochs))

            self.logger.debug('Preparing epoch')
            self.prepare_next_epoch(model, data, epoch)

            bar = self.create_progress_bar('loss')
            train_losses = []  # used to calculate the epoch train loss
            recent_train_losses = []  # used to calculate the display loss

            self.logger.debug('Training')
            self.logger.debug('{} minibatches with size {}'.format(self.get_n_batches(), self.batchsize))

            for _ in bar(range(int(self.get_n_batches()))):
                # self.global_step += self.batchsize
                train_examples = self.get_next_batch(model, data)
                batch_loss = 0

                batch_steps = int(np.ceil(len(train_examples) / (self.batchsize / self.gradient_accumulation_steps)))
                for i in range(batch_steps):
                    step_size = self.batchsize // self.gradient_accumulation_steps
                    step_examples = train_examples[i * step_size:(i + 1) * step_size]
                    step_loss = self.get_loss(model, step_examples, self.adapter_task)
                    step_loss = step_loss / self.gradient_accumulation_steps
                    step_loss.backward()
                    batch_loss += step_loss.item()

                if self.config.get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.bert.parameters(), self.config.get('max_grad_norm'))
                    # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                optimizer.step()
                scheduler.step(epoch)
                optimizer.zero_grad()
                global_step += 1

                self.tensorboard.add_scalars('scores', {'train_loss': batch_loss}, global_step=global_step)

                recent_train_losses = ([batch_loss] + recent_train_losses)[:20]
                train_losses.append(recent_train_losses[0])
                bar.dynamic_messages['loss'] = np.mean(recent_train_losses)

            self.logger.info('train loss={:.6f}'.format(np.mean(train_losses)))

            if self.config.get('evaluate_dev', True):
                self.logger.info('Now calculating validation score')
                valid_score, valid_score_other_measures = evaluation.start(model, data, valid_only=True)
                valid_score = list(valid_score.values())[0]  # get only the dev split (there wont be any other split)
                valid_score_other_measures = list(valid_score_other_measures.values())[0]

                self.tensorboard.add_scalar('valid_score', valid_score, global_step=global_step)
                self.tensorboard.add_scalars('scores',
                                             dict([('valid_' + k, v) for k, v, in valid_score_other_measures.items()]),
                                             global_step=global_step)
                for key, value in valid_score_other_measures.items():
                    self.tensorboard.add_scalar(key, value, global_step=global_step)

                self.state.record(model.bert, optimizer if self.chkpt_optimizer else None, valid_score, self.backup_checkpoint_every)
            else:
                self.logger.info('Not validating dev. Setting score to epoch')
                self.state.record(model.bert, optimizer if self.chkpt_optimizer else None, epoch, self.backup_checkpoint_every)

        return self.state.best_epoch, self.state.best_score

    def restore_best_epoch(self, model, data, evaluation, validate=True):
        self.logger.info('Restoring the weights of the best epoch {} with score {}'.format(
            self.state.best_epoch, self.state.best_score
        ))
        self.state.load(model.bert, optimizer=None, weights='best')
        if validate:
            self.logger.info('Now calculating validation score (to verify the restoring success)')
            valid_score = list(evaluation.start(model, data, valid_only=True)[0].values())[0]
            self.logger.info('Score={:.4f}'.format(valid_score))
            return valid_score
        else:
            self.logger.info('Skipping calculating validation score (to verify the restoring success)!')
            return None

    def remove_checkpoints(self):
        """Removes all the persisted checkpoint data that was generated during training for restoring purposes"""
        self.state.clear()

    def prepare_training(self, model, data):
        """Prepares data that is used in all training epochs, i.e., positive pairs.
        Sets the value of train_positive_instances

        :param model:
        :param data:
        :return:
        """
        self.data = data

    def get_batch_indices(self, dataset_id, size):
        """Keeps track of random indices for processing the train splits. Ensures that all instances of a train dataset
        are used throughout the training independent of how large epochs are.

        :return: Next set of random indices for a given dataset.
        """
        indices = self.train_random_indices[dataset_id]
        batch_indices = indices[:size]
        missing_indices = size - len(batch_indices)

        if missing_indices > 0:
            # indices are empty, create a new random permutation
            indices = np.random.permutation(len(self.train_positive_instances[dataset_id])).tolist()
            batch_indices += indices[:missing_indices]
            remaining_indices = indices[missing_indices:]
        else:
            remaining_indices = indices[size:]

        self.train_random_indices[dataset_id] = remaining_indices
        return batch_indices

    def get_random_answers(self, dataset_id, n):
        neg_answers = self.train_negative_answers[dataset_id]
        batch_neg_answers = neg_answers[:n]
        missing_answers = n - len(batch_neg_answers)

        if missing_answers > 0:
            # indices are empty, create a new random permutation
            if self.use_all_negative_sampling:
                negative_samples = list(self.data.datas_train[dataset_id].archive.answers)
                negative_samples += list(self.data.datas_train[dataset_id].archive.additional_answers)
            else:
                negative_samples = list(self.data.datas_train[dataset_id].archive.train.answers)
            np.random.shuffle(negative_samples)
            batch_neg_answers += negative_samples[:missing_answers]
            remaining_neg_answers = negative_samples[missing_answers:]
        else:
            remaining_neg_answers = neg_answers[n:]

        self.train_negative_answers[dataset_id] = remaining_neg_answers
        return batch_neg_answers

    def prepare_next_epoch(self, model, data, epoch):
        """Prepares the next epoch

        """
        self.batch_i = 0

    def get_n_batches(self):
        """

        :return: the number of batches in one epoch (this depends on the training strategy that is used)
        """
        raise NotImplementedError()

    def get_next_batch(self, model, data):
        """Return the training data for the next batch. If we configured multiple training datasets, each positive
        example will only be paired with negative examples from the same training set (see self.train_dataset_ids)

        :return: list[InputExample] or list[(InputExample, InputExample, label)]
        :rtype: list
        """
        raise NotImplementedError()
