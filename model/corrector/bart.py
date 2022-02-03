from torch import nn
from model.utils import _pretrained2Lm
from model.copy_transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline



class BartCorrector(nn.Module):
    def __init__(self, args):
        super(BartCorrector, self).__init__()
        self.args = args
        self.encoder = BartForConditionalGeneration.from_pretrained(args.pretrained_model_name)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask, encoder_pinyin_ids = inputs['input_ids'], inputs.get('token_type_ids', None), inputs.get('attention_mask', None), inputs.get('pinyin_ids', None)
        position_ids = inputs.get('position_ids', None)
        decoder_input_ids = inputs.get('decoder_input_ids', None)
        decoder_pinyin_ids = inputs.get('pinyin_ids', None)

        if not self.args.encoder_pinyin_ids:
            encoder_pinyin_ids = None
        if not self.args.decoder_pinyin_ids:
            decoder_pinyin_ids = None
        if self.args.decode_on_pinyin:
            decoder_input_ids = inputs.get('pinyin_ids', None)
            decoder_pinyin_ids = None
        # print("decoder_input_ids:",decoder_input_ids)
        token_logits = self.encoder(input_ids=input_ids, decoder_input_ids=decoder_input_ids, pinyin_ids=encoder_pinyin_ids, decoder_pinyin_ids=decoder_pinyin_ids).logits
        return token_logits




