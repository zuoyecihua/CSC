import torch
from torch import nn
from model.corrector.pinyin import ErrPinyinCorrector
from data.call_func import _get_tokenizer, sighan_error_correct_prompt, text2pinyin_similar
from data.utils import pad_0,pad_positions
from model.copy_transformers import AutoModelForMaskedLM, BertForMaskedLM #suport pinyin ids
from data.utils import build_candidates_dict, build_candidates_for_all

class DetecSelectPipeline(nn.Module):
    def __init__(self, args):
        super(DetecSelectPipeline, self).__init__()
        self.args = args
        self.detector = BertForMaskedLM.from_pretrained(args['pretrained_model_name'])

        if args['share_param']:
            self.selector = self.detector
        else:
            self.selector = BertForMaskedLM.from_pretrained(args['pretrained_model_name'])
        self.tokenizer = _get_tokenizer()
        self.detector.resize_token_embeddings(len(self.tokenizer))
        self.selector.resize_token_embeddings(len(self.tokenizer))

        self.selector_linear = nn.Linear(self.args.hidden_size, 2)
        self.word2candidates  = build_candidates_for_all()
        self.pinyin = self.args.pinyin
        self.position = self.args.position


    def forward(self, inputs):
        self.device = inputs['input_ids'].device
        raw_items = inputs['raw_items']
        detector_encoded = self.detector_forward(inputs, self.pinyin, self.position)
        new_items = [self.process_detect_logits_for_single_item(raw_items[i], detector_encoded['logits'][i]) for i in range(len(detector_encoded['logits']))]
        new_inputs = self.sighan_error_correct_prompt(new_items)

        for k in new_inputs.keys():
            if isinstance(new_inputs[k], torch.Tensor):
                new_inputs[k] = new_inputs[k].to(self.device)
        selector_encoded = self.selector_forward(new_inputs, self.pinyin)

        select_logits = self.selector_linear(selector_encoded['hidden_states']) * new_inputs['select_masks'][:, :, None]
        return select_logits, detector_encoded, selector_encoded, new_inputs['select_labels'], new_inputs['correct_labels']


    def detector_forward(self, inputs, pinyin, position):
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids',None)
        attention_mask = inputs.get('attention_mask', None)
        pinyin_ids = inputs.get('pinyin_ids', None) if pinyin else None
        position_ids = inputs.get('position_ids', None) if position else None
        encoded = self.detector(input_ids=input_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask, pinyin_ids=pinyin_ids,
                             position_ids=position_ids, output_hidden_states=False)
        return encoded

    def selector_forward(self, inputs, pinyin):
        input_ids = inputs['input_ids']
        token_type_ids = inputs.get('token_type_ids',None)
        attention_mask = inputs.get('attention_mask', None)
        pinyin_ids = inputs.get('pinyin_ids', None) if pinyin else None
        position_ids = inputs.get('position_ids')

        encoded = self.selector(input_ids=input_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask, pinyin_ids=pinyin_ids,
                             position_ids=position_ids, output_hidden_states=True)

        return {'logits':encoded['logits'],'hidden_states':encoded['hidden_states'][-1]}






    # def process_detect_logits_for_batch(self, batch_logits):
    #     items = [self.process_detect_logits_for_single_item(batch_logits[i]) for i in range(len(batch_logits))]
    #     inputs = sighan_error_correct_prompt(items)
    #     for k in inputs.keys():
    #         if isinstance(inputs[k],torch.Tensor):
    #             inputs[k] = inputs[k].to(self.device)
    #     return inputs

    def process_detect_logits_for_single_item(self, raw_item, detec_logits):
        detec_logits_probablity = detec_logits.softmax(dim=1)
        detec_predicts = detec_logits_probablity.argmax(dim=-1)
        detec_tokens = self.tokenizer.convert_ids_to_tokens(detec_predicts)[1:-1]

        id, text, label = raw_item
        tokens = list(text)
        positions = list(range(len(tokens) + 1))
        select_masks = list([0 for x in range(len(tokens) + 1)])

        label_splits = label.split(",")

        all_candidates = []
        for i in range(len(text) - 1):
            text = text.lower()
            j = 0
            minus = j if i > 0 else 0
            added = j if i < len(text) - 1 else 0
            logit = detec_logits_probablity[i + 1][detec_predicts[i + 1]].item()
            # print(detec_tokens[i],logit)
            if text[i] != detec_tokens[i] or text[i - minus] != detec_tokens[i - minus] or text[i + added] != \
                    detec_tokens[i + added] or logit < 0.8:
                if detec_tokens[i] == '，':
                    continue
                pos = i
                origin_token = tokens[pos]
                candidates = self.word2candidates[origin_token]
                cids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(candidates))
                clogits = detec_logits_probablity[pos + 1][cids].tolist()
                good = [x[1] for x in sorted(list(zip(clogits, candidates)), key=lambda x: -x[0])[:5]]
                candidates = list(set("".join(good) + origin_token + detec_tokens[i]))
                all_candidates += candidates
                positions += [pos + 1] * len("".join(candidates))  # tokenizer会增加cls，所以pos + 1
        input_text = ["".join(tokens), "".join(all_candidates)]

        tokens = list(text)
        for i in range(1, len(label_splits), 2):
            if i + 1 >= len(label_splits):
                break
            pos = int(label_splits[i]) - 1
            token = label_splits[i + 1].strip()
            tokens[pos] = token

        label_text = ["".join(tokens), "".join(all_candidates)]

        select_labels = list([0 for x in range(len(tokens) + 1)])
        select_labels.append(0)
        positions += [len(positions)]
        select_masks.append(0)
        # assert len(positions[-(len(all_candidates)+1):-1]) == len(all_candidates)
        all_candidate_labels = []
        for pos, candate in zip(positions[-(len(all_candidates) + 1):-1], all_candidates):
            label_token = label_text[0][pos - 1]
            all_candidate_labels.append(label_token)
            if candate == label_token:
                select_label = 1
            else:
                select_label = 0
            select_masks.append(1)
            select_labels.append(select_label)
        # print(len(select_labels),len(select_masks),len(positions))
        label_text[1] = "".join(all_candidate_labels)
        select_labels.append(0)
        positions += [len(positions)]  # for sep
        select_masks.append(0)
        modified_item = (id, input_text, positions, label_text, select_masks, select_labels)
        return modified_item

    def sighan_error_correct_prompt(self, batch):
        _tokenizer = _get_tokenizer()
        max_len = 510
        sentence_ids, input_texts, positions, label_texts, select_masks, select_labels = zip(*batch)

        inputs = _tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors="pt", truncation="only_second",
                                              max_length=max_len)

        inputs['correct_labels'] = \
        _tokenizer.batch_encode_plus(label_texts, padding=True, return_tensors="pt", truncation="only_second",
                                     max_length=max_len)['input_ids']
        inputs['sentence_ids'] = sentence_ids
        seq_len = inputs['input_ids'].shape[1]
        inputs['position_ids'] = torch.LongTensor(pad_positions(positions, seq_len))  # prompt的答案位置
        inputs['select_masks'] =  torch.LongTensor(pad_0(select_masks, seq_len))
        inputs['select_labels'] = torch.LongTensor(pad_0(select_labels, seq_len))

        # print("inputs['position_ids']", inputs['position_ids'].shape)
        pinyin_ids = [(_tokenizer.convert_tokens_to_ids(text2pinyin_similar(a)) + [
            _tokenizer.sep_token_id] + _tokenizer.convert_tokens_to_ids(text2pinyin_similar(b)))[:seq_len - 2] for a, b
                      in input_texts]

        inputs['pinyin_ids'] = torch.LongTensor([[_tokenizer.cls_token_id] + ids + [_tokenizer.sep_token_id] + [
            _tokenizer.pad_token_id] * (seq_len - len(ids) - 2) for ids in pinyin_ids])
        return inputs
