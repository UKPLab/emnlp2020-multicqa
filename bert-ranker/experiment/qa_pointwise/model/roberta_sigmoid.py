import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers.modeling_roberta import RobertaForSequenceClassification, RobertaClassificationHead

from experiment.bert_utils import BertWrapperModel
from experiment.qa.model import BaseModel
from torch import nn

class RobertaSigmoid(BaseModel):
    _MODEL = RobertaModel

    def __init__(self, from_pretrained, model_name=None, cache_dir=None, config=None, num_labels=1):
        super(RobertaSigmoid, self).__init__(from_pretrained, model_name=model_name, cache_dir=cache_dir, config=config)
        assert num_labels == 1
        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.lin_layer = nn.Linear(self.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None, tasks=None):
        # we set the token type ids to zero
        # -> https://github.com/huggingface/transformers/issues/1443#issuecomment-581019419
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids=token_type_ids * 0.0, attention_mask=attention_mask, position_ids=None,
            head_mask=head_mask
        )[:2]

        # sent_encoding = pooled_output
        sent_encoding = encoded_layers[:, 0, :]
        # sent_encoding = self.dropout(sent_encoding)
        sent_encoding = self.lin_layer(sent_encoding)
        return self.sigmoid(sent_encoding)



class RobertaSigmoidModel(BertWrapperModel):
    _MODEL_CLASS = RobertaSigmoid
    _TOKENIZER_CLASS = RobertaTokenizer
    _CONFIG_CLASS = RobertaConfig


component = RobertaSigmoidModel
