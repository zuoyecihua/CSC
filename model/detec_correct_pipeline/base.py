import torch
from torch import nn
from model.corrector.pinyin import ErrPinyinCorrector
from data.call_func import _get_tokenizer,sighan_error_correct_prompt

class DetecCorrectPipeline(nn.Module):
    def __init__(self, args, word2candidate):
        super(DetecCorrectPipeline, self).__init__()
        self.detector = ErrPinyinCorrector(args)
        self.selector = ErrPinyinCorrector(args)
        self.args = args
        self.word2candidate = word2candidate
        self.tokenizer = _get_tokenizer()


    def forward(self, inputs):

        batch_logits = self.detector(inputs)
        new_inputs = self.process_detect_logits_for_batch(batch_logits)
        prompt_logits =  self.selector(new_inputs)
        return batch_logits, prompt_logits




    def process_detect_logits_for_batch(self, batch_logits):
        items = [self.process_detect_logits_for_single_item(batch_logits[i]) for i in range(len(batch_logits))]
        inputs = sighan_error_correct_prompt(items)
        for k in inputs.keys():
            if isinstance(inputs[k],torch.Tensor):
                inputs[k] = inputs[k].to(self.device)
        return inputs



    def process_detect_logits_for_single_item(self, raw_item , detec_logits_probablity):
        detec_predicts = detec_logits_probablity.argmax(dim=-1)
        detec_tokens = self.tokenizer.convert_ids_to_tokens(detec_predicts)[1:-1]

        id, text, label = raw_item
        tokens = list(text)
        positions = list(range(len(tokens) + 1))
        label_splits = label.split(",")

        all_candidates = []
        for i in range(len(text) - 1):
            text = text.lower()
            j = 0
            minus = j if i > 0 else 0
            added = j if i < len(text) - 1 else 0
            logit = detec_logits_probablity[i + 1][detec_predicts[i + 1]].item()

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

        positions += [len(positions)]  # for sep
        modified_item = (id, input_text, positions, label_text)
        return modified_item

