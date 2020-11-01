# from transformers import BertModel, BertTokenizer

from torch import nn
from transformers import BertTokenizer
import torch
from experiment.bert_models.modeling_bert import BertModel, BertConfig
from experiment.bert_utils import BertWrapperModel
from experiment.qa.model import BaseModel

class BertSigmoid(BaseModel):
    _MODEL = BertModel

    def __init__(self, from_pretrained, model_name=None, cache_dir=None, config=None, num_labels=1):
        super(BertSigmoid, self).__init__(from_pretrained, model_name=model_name, cache_dir=cache_dir, config=config)
        assert num_labels == 1
        self.num_labels = num_labels
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.lin_layer = nn.Linear(self.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None, tasks=None):
        encoded_layers, pooled_output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=None,
            head_mask=head_mask, tasks=tasks
        )[:2]

        # sent_encoding = pooled_output
        sent_encoding = encoded_layers[:, 0, :]
        # sent_encoding = self.dropout(sent_encoding)
        sent_encoding = self.lin_layer(sent_encoding)
        return self.sigmoid(sent_encoding)

    def average_standard_bert_output(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=None,
            head_mask=head_mask
        )[:2]

        attention_mask = (attention_mask == 0).float().to(encoded_layers.device)

        length = torch.sum(attention_mask, dim=1)

        attention_mask = attention_mask[:,:,None].repeat((1,1, encoded_layers.size()[-1]))

        encoded_layers = encoded_layers * attention_mask

        layer_sum = torch.sum(encoded_layers, dim=1)
        mean = layer_sum / length[:,None].repeat(1,encoded_layers.size()[-1])
        return mean


class BertSigmoidModel(BertWrapperModel):
    _MODEL_CLASS = BertSigmoid
    _TOKENIZER_CLASS = BertTokenizer
    _CONFIG_CLASS = BertConfig

component = BertSigmoidModel
