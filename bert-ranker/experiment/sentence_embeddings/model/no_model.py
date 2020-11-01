from experiment.bert_utils import BertWrapperModel

class NoModel(BertWrapperModel):
    def build(self, data):
        pass

component = NoModel

