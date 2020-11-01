from torch.nn import Module

class BaseModel(Module):
    """
    Set _MODEL to the Transformer Model used in the model of a child of this class
    """
    _MODEL = None

    def __init__(self, from_pretrained, model_name=None, cache_dir=None, config=None):
        """

        :param from_pretrained: load pretrained model or use a given config
        :param model_name: needed if from_pretrained
        :param cache_dir: needed if from_pretrained
        :param config: needed if not from_pretrained. Instance of transformers.PretrainedConfig
        """
        super(BaseModel, self).__init__()
        if from_pretrained:
            self.bert = self._MODEL.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            self.bert = self._MODEL(config)
        self.config = self.bert.config