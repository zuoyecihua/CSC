from torch import nn
from model.utils import _pretrained2Lm
from transformers import AutoModelForMaskedLM,BertForMaskedLM


class ErrCorrector(nn.Module):
    def __init__(self, args):
        super(ErrCorrector, self).__init__()
        self.args = args
        self.encoder = _pretrained2Lm.get(args['pretrained_model_name'], AutoModelForMaskedLM).from_pretrained(args['pretrained_model_name'])


    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs.get('token_type_ids', None), inputs.get('attention_mask', None)
        if self.args.only_ids:
            encoded = self.encoder(input_ids=input_ids)
        else:
            encoded = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        token_logits = encoded['prediction_logits']
        #hiddens = encoded['hidden_states']
        return token_logits




