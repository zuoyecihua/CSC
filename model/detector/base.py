import torch
from torch import nn
from model.utils import _pretrained2loader
from transformers import BertTokenizer, AlbertModel, AutoModel,BertModel, BertForPreTraining


class ErrDetector(nn.Module):
    def __init__(self, args):
        super(ErrDetector, self).__init__()
        self.args = args
        self.encoder = _pretrained2loader.get(args['pretrained_model_name'], AutoModel).from_pretrained(args['pretrained_model_name'])
        self.cls = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs.get('token_type_ids', None), inputs.get('attention_mask', None)
        encoded = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        error_logits = self.cls(encoded['last_hidden_state'])
        return error_logits




