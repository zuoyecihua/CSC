from torch import nn
from model.utils import _pretrained2Lm
from model.copy_transformers import AutoModelForMaskedLM, BertForMaskedLM


class ContrastCorrector(nn.Module):
    def __init__(self, args):
        super(ContrastCorrector, self).__init__()
        self.args = args
        self.encoder = BertForMaskedLM.from_pretrained(args['pretrained_model_name'])
        self.cls_embeddings = self.encoder.bert.embeddings.word_embeddings


    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask, pinyin_ids = inputs['input_ids'], inputs.get('token_type_ids', None),inputs.get(
            'attention_mask', None), inputs.get('pinyin_ids', None)
        candidates = inputs['candidates']
        position_ids = inputs.get('position_ids', None)
        if not self.args.pinyin:
            pinyin_ids = None


        encoded = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                               pinyin_ids=pinyin_ids, position_ids=position_ids)

        token_logits = encoded['logits']
        return token_logits




