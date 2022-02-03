import torch
from torch.utils.data import Dataset
import os
from .utils import build_candidates_dict
import re
class SigDataset(Dataset):
    def __init__(self, args):
        text_path, label_path = args.text_path, args.label_path
        print(os.getcwd())
        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            sentence_ids, texts = zip(*[x.strip().split("\t") for x in lines])
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [x.strip() for x in f.readlines()]
        self.ids = sentence_ids
        self.labels = labels
        self.texts = texts
        assert len(self.labels) == len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        sentence_id = self.ids[item]

        return sentence_id, text, label

    def __len__(self):
        return len(self.ids)


class SigPromptDataset(SigDataset):
    def __init__(self, args):
        super(SigPromptDataset, self).__init__(args)
        self.word2candidates = build_candidates_dict()
        self.prepare_datasets()


    def prepare_datasets(self):
        print("prepare prompt dataset")
        self.prompt_input_texts, self.prompt_label_texts = [], []
        self.prompt_positions = []
        for text, label in zip(self.texts,self.labels):
            tokens = list(text)
            positions = list(range(len(tokens)+1))
            label_splits = label.split(",")
            all_candidates = []
            for i in range(1, len(label_splits), 2):
                if i + 1 >= len(label_splits):
                    break
                pos = int(label_splits[i]) - 1
                # token = label_splits[i + 1].strip()
                origin_token = tokens[pos]
                candidate_string = origin_token + "".join(self.word2candidates[origin_token])
                candidates = list(set(re.sub(r"\s", "", candidate_string)))

                all_candidates += candidates
                positions += [pos+1] * len("".join(candidates)) #tokenizer会增加cls，所以pos + 1
                # tokens[pos] = "_"
            input_text = ["".join(tokens), "".join(all_candidates)]
            positions += [len(positions)] #for sep
            for i in range(1, len(label_splits), 2):
                if i + 1 >= len(label_splits):
                    break
                pos = int(label_splits[i]) - 1
                token = label_splits[i + 1].strip()
                tokens[pos] = token
            label_text = ["".join(tokens) ,"".join(all_candidates)]
            self.prompt_input_texts.append(input_text)
            self.prompt_label_texts.append(label_text)
            self.prompt_positions.append(positions)
        print("prepare prompt dataset completed!")



    def __getitem__(self, item):
        label_text = self.prompt_label_texts[item]
        input_text = self.prompt_input_texts[item]
        positions = self.prompt_positions[item]
        # print(positions)
        # print(input_text)
        # print(label_text)
        # print(len(input_text), len(positions))
        assert len(label_text[0]) + len(label_text[1]) == len(input_text[0] ) + len(input_text[1])
        # print(positions)
        # print(input_text)
        # print(len(input_text), len(positions))
        # assert len(positions) == len(input_text) + 2

        sentence_id = self.ids[item]
        return sentence_id, input_text, positions, label_text

class SigPromptDatasetFromPkl:
    def __init__(self, args):
        self.word2candidates = build_candidates_dict()
        self.items = torch.load(args.text_path)

    def __getitem__(self, item):

        return self.items[item]
    def __len__(self):
        return len(self.items)