import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self, input_size, down_sample=None, non_linearity='relu', init_bert_weights=True):
        super().__init__()
        self.input_size = input_size

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        # TODO give more options than just relu, or pass the non_linearity directly, not as a string
        if non_linearity.lower() == 'relu':
            self.non_linearity = nn.ReLU()
        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # attention layer that learns the usefulness of the different adapters. Is only trained in the later steps
        self.adapter_attention = nn.Linear(self.down_sample, 1)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_attention.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)

    def forward(self, x, residual_input):
        down = self.adapter_down(x)
        attention = self.adapter_attention(down)
        up = self.adapter_up(down)
        output = up + residual_input

        return output, attention, down, up

    # This is copied from the BERT model so that this is a self containing class. This unfortunately introduces code
    # copying so it might be better to pass the BERT model here TODO
    @staticmethod
    def init_bert_weights(module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertAdapterAttention(nn.Module):
    def __init__(self, config):
        super(BertAdapterAttention, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions

        self.dense_size = int(config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if not self.config.fusion_config['query'] and \
                not self.config.fusion_config['key'] and \
                not self.config.fusion_config['value']:
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config.fusion_config['query']:
            self.query = nn.Linear(int(config.hidden_size), self.dense_size)
            self.query.apply(Adapter.init_bert_weights)

        if self.config.fusion_config['key']:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(Adapter.init_bert_weights)

        if self.config.fusion_config['value']:
            self.value = nn.Linear(int(config.hidden_size), int(config.hidden_size), bias=False)
            self.value.apply(Adapter.init_bert_weights)
            if self.config.fusion_config['value_initialized']:
                self.value.weight.data = (
                        torch.zeros(int(config.hidden_size), int(config.hidden_size)) + 0.000001
                ).fill_diagonal_(1.0)

        if self.config.fusion_config['temperature']:
            self.T = 50.0
        else:
            self.T = 1.0

        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual, attention_mask=None):
        if self.config.fusion_config['residual_before']:
            value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)

        if self.config.fusion_config['query']:
            query_layer = self.query(query)
        else:
            query_layer = query

        if self.config.fusion_config['key']:
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config.fusion_config['value'] and self.config.fusion_config['value_before_softmax']:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores = self.dropout(attention_scores)

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)

        # attention_probs = torch.zeros_like(attention_probs)
        # attention_probs[:, :, 0] = 1.0

        if not self.training:
            self.recent_attention = attention_probs.detach().cpu().numpy()

        self.T = max(self.T - self.reduction, 1.0)

        # use the value layer or not TODO this is currently hardcoded
        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)
        # context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value), dim=2)

        if self.config.fusion_config['value'] and not self.config.fusion_config['value_before_softmax']:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer

        if not self.config.fusion_config['residual_before']:
            context_layer += residual

        return context_layer


# This is copied from the BERT model so that this is a self containing class. This unfortunately intorduces code
# copying so it might be better to pass the BERT model here, but who cares...
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
