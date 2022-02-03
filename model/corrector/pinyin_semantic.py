from torch import nn
from model.utils import _pretrained2Lm
from model.copy_transformers import AutoModelForMaskedLM, BertForMaskedLM #suport pinyin ids


class PinyinSemanticCorrector(nn.Module):
    def __init__(self, args):
        super(PinyinSemanticCorrector, self).__init__()
        self.args = args
        self.encoder = BertForMaskedLM.from_pretrained(args['pretrained_model_name'])

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids', None)
        attention_mask = inputs.get('attention_mask', None)
        pinyin_ids = inputs.get('pinyin_ids', None) if self.args.pinyin else None
        position_ids = inputs.get('position_ids', None) if self.args.position else None

        encoded1 = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, pinyin_ids=pinyin_ids, position_ids=None)
        encoded2 = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                               pinyin_ids=None, position_ids=None)
        return encoded1['logits'] + encoded2['logits']




