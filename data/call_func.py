import os.path

from model.copy_transformers import BertTokenizer
import torch
from tqdm import tqdm
import pypinyin
from .utils import text2pinyin_similar, pad_positions
from utils.args_util import load_args
from transformers.models.bart.modeling_bart import shift_tokens_right
_tokenizer = None
_token_config = load_args("./configs/util.yaml").token_config

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(_token_config)
        import types
        def my_tokenize(self, text):
            # print("use my tokenizer !!!")
            return list(text.lower())
        _tokenizer.tokenize = types.MethodType(my_tokenize, _tokenizer)
        if os.path.exists("./tests/tokens.pkl") :
            print("load saved tokens")
            tokens = torch.load("./tests/tokens.pkl")
        else:
            tokens = set()
            for line in tqdm(open("resources/merged/extend_training_input_shuf.txt","r", encoding="utf-8")):
                tokens.update(list(line))
                tokens.update(pypinyin.lazy_pinyin(list(line)))
            torch.save(tokens, "./tests/tokens.pkl")
        _tokenizer.add_tokens(sorted(list(tokens)))
    return _tokenizer


def sighan_error_judge(batch):
    _tokenizer = _get_tokenizer()
    sentence_ids, texts, labels = zip(*batch)
    inputs = _tokenizer.batch_encode_plus(texts, padding=True, return_tensors="pt")
    inputs['judge_labels'] = torch.LongTensor([(1 if len(label.split(",")) > 2 else 0) for label in labels])
    inputs['sentence_ids'] = sentence_ids
    # print(_tokenizer.batch_decode(inputs['input_ids']))
    return inputs

def sighan_error_detec_csc(batch):
    _tokenizer = _get_tokenizer()
    sentence_ids, texts, labels = zip(*batch)
    inputs = _tokenizer.batch_encode_plus(texts, padding=True, return_tensors="pt")
    inputs['judge_labels'] = torch.LongTensor([(1 if len(label.split(",")) > 2 else 0) for label in labels])
    inputs['detec_labels'] = torch.zeros_like(inputs['input_ids'], dtype=torch.long)
    inputs['correct_labels'] = inputs['input_ids'].detach().clone()  # 复制tensor

    seq_len = inputs['input_ids'].shape[1]
    pinyin_ids = [ _tokenizer.convert_tokens_to_ids(text2pinyin_similar(text)) for text in texts]
    inputs['pinyin_ids'] = torch.LongTensor([[_tokenizer.cls_token_id] + ids + [_tokenizer.sep_token_id] + [_tokenizer.pad_token_id] * (seq_len - len(ids) - 2) for ids in pinyin_ids])

    for sample_id, label in enumerate(labels):
        label_splits = label.split(",")
        for i in range(1,len(label_splits), 2):
            if i + 1 >= len(label_splits):
                break
            pos = int(label_splits[i]) - 1 + 1 # add 1 because of [CLS]
            token = label_splits[i+1].strip()
            token_id = _tokenizer.convert_tokens_to_ids([token])[0]

            # print(texts[sample_id])
            # print(sample_id, pos,token, token_id, sentence_ids[sample_id], inputs['detec_labels'].shape)

            if pos > len(inputs['detec_labels'][0]):
                print(sample_id, pos, token_id, sentence_ids[sample_id], inputs['detec_labels'].shape)
                assert 1==2
            inputs['detec_labels'][sample_id, pos] = 1
            inputs['correct_labels'][sample_id, pos] = token_id

            # print()
            # print(inputs['correct_labels'][sample_id, pos], token_id, inputs['input_ids'][sample_id, pos])

    inputs['sentence_ids'] = sentence_ids
    inputs['raw_items'] = batch
    pad_token_id, decoder_start_token_id = 0, 102
    inputs['decoder_input_ids'] = shift_tokens_right(inputs['correct_labels'], pad_token_id, decoder_start_token_id)
    inputs['decoder_pinyin_ids'] = shift_tokens_right(inputs['pinyin_ids'], pad_token_id, decoder_start_token_id)
    return inputs

def sighan_error_detec_csc_mask_errors_random(batch):
    _tokenizer = _get_tokenizer()
    sentence_ids, texts, labels = zip(*batch)
    inputs = _tokenizer.batch_encode_plus(texts, padding=True, return_tensors="pt")
    inputs['judge_labels'] = torch.LongTensor([(1 if len(label.split(",")) > 2 else 0) for label in labels])
    inputs['detec_labels'] = torch.zeros_like(inputs['input_ids'], dtype=torch.long)
    inputs['correct_labels'] = inputs['input_ids'].detach().clone()  # 复制tensor

    seq_len = inputs['input_ids'].shape[1]
    pinyin_ids = [ _tokenizer.convert_tokens_to_ids(text2pinyin_similar(text)) for text in texts]
    inputs['pinyin_ids'] = torch.LongTensor([[_tokenizer.cls_token_id] + ids + [_tokenizer.sep_token_id] + [_tokenizer.pad_token_id] * (seq_len - len(ids) - 2) for ids in pinyin_ids])

    for sample_id, label in enumerate(labels):
        label_splits = label.split(",")
        for i in range(1,len(label_splits), 2):
            if i + 1 >= len(label_splits):
                break
            pos = int(label_splits[i]) - 1 + 1 # add 1 because of [CLS]
            token = label_splits[i+1].strip()
            token_id = _tokenizer.convert_tokens_to_ids([token])[0]

            # print(texts[sample_id])
            # print(sample_id, pos,token, token_id, sentence_ids[sample_id], inputs['detec_labels'].shape)

            if pos > len(inputs['detec_labels'][0]):
                print(sample_id, pos, token_id, sentence_ids[sample_id], inputs['detec_labels'].shape)
                assert 1==2
            inputs['detec_labels'][sample_id, pos] = 1
            inputs['correct_labels'][sample_id, pos] = token_id
            if torch.rand(1).item() < 0.5:
                inputs['input_ids'][sample_id, pos] = _tokenizer.mask_token_id

            # print()
            # print(inputs['correct_labels'][sample_id, pos], token_id, inputs['input_ids'][sample_id, pos])
    inputs['input_ids'][torch.rand(inputs['input_ids'].shape) < 0.1] = _tokenizer.mask_token_id
    inputs['sentence_ids'] = sentence_ids
    inputs['raw_items'] = batch
    pad_token_id, decoder_start_token_id = 0, 102
    inputs['decoder_input_ids'] = shift_tokens_right(inputs['correct_labels'], pad_token_id, decoder_start_token_id)
    inputs['decoder_pinyin_ids'] = shift_tokens_right(inputs['pinyin_ids'], pad_token_id, decoder_start_token_id)
    return inputs

def sighan_error_judge_prompt(batch):
    _tokenizer = _get_tokenizer()
    sentence_ids, input_texts, label_texts, label_idxes = zip(*batch)
    inputs = _tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors="pt")

    inputs['label_ids'] = _tokenizer.batch_encode_plus(label_texts, padding=True, return_tensors="pt")['input_ids']
    inputs['sentence_ids'] = sentence_ids
    inputs['answer_idxes'] = torch.LongTensor(label_idxes).unsqueeze(1)  #prompt的答案位置
    return inputs


def sighan_error_correct_prompt(batch):
    _tokenizer = _get_tokenizer()
    sentence_ids, input_texts, positions, label_texts = zip(*batch)


    inputs = _tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors="pt", truncation="only_second", max_length=510)

    inputs['correct_labels'] = _tokenizer.batch_encode_plus(label_texts, padding=True, return_tensors="pt", truncation="only_second", max_length=510)['input_ids']
    inputs['sentence_ids'] = sentence_ids
    seq_len = inputs['input_ids'].shape[1]
    inputs['position_ids'] = torch.LongTensor(pad_positions(positions, seq_len)) #prompt的答案位置

    # print("inputs['position_ids']", inputs['position_ids'].shape)
    pinyin_ids = [(_tokenizer.convert_tokens_to_ids(text2pinyin_similar(a)) + [
        _tokenizer.sep_token_id] + _tokenizer.convert_tokens_to_ids(text2pinyin_similar(b)))[:seq_len-2] for a, b in input_texts]

    inputs['pinyin_ids'] = torch.LongTensor([[_tokenizer.cls_token_id] + ids + [_tokenizer.sep_token_id] + [
        _tokenizer.pad_token_id] * (seq_len - len(ids) - 2) for ids in pinyin_ids])
    inputs['raw_items'] = batch
    return inputs

_call_funcs = {
    "sighan_error_judge": sighan_error_judge,
    "sighan_error_detec_csc": sighan_error_detec_csc,
    "sighan_error_judge_prompt": sighan_error_judge_prompt,
    "sighan_error_correct_prompt": sighan_error_correct_prompt,
    "sighan_error_detec_csc_mask_errors_random": sighan_error_detec_csc_mask_errors_random
}

def get_call_func_by_name(key):
    return _call_funcs[key]
