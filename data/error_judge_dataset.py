from torch.utils.data import Dataset
import os

class ErrJudgeDataset(Dataset):
    def __init__(self, args):
        text_path, label_path = args.text_path, args.label_path
        print(os.getcwd())
        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            sentence_ids, texts = zip(*[x.strip().split("\t") for x in lines])
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [x.strip() for x in f.readlines()]

        corrected_ids, corrected_sentences, corrected_labels = [], [], []
        for id_, sentence,label in zip(sentence_ids, texts, labels):
            id_ = id_ + "_corrected"
            new_sentence = list(sentence)
            label_tokens = label.split(",") [1:]
            for i  in range(0, len(label_tokens),2):
                if len(label_tokens[i:i+2]) != 2:
                    continue
                pos, label_character = label_tokens[i:i+2]
                new_sentence[int(pos)-1] = label_character.strip()
            new_sentence = "".join(new_sentence)
            corrected_ids.append(id_)
            corrected_sentences.append(new_sentence)
            corrected_labels.append(",".join([label.split(",")[0], '0']))


        def merge_list(la,lb):
            l = []
            for i in range(len(la)):
                l.append(la[i])
                l.append(lb[i])
            return l

        self.ids = merge_list(sentence_ids, corrected_ids)
        self.labels = merge_list(labels,corrected_labels)
        self.texts = merge_list(texts,corrected_sentences)


    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        sentence_id = self.ids[item]
        return sentence_id, text, label

    def __len__(self):
        return len(self.ids)

class ErrJudgePromptDataset(ErrJudgeDataset):
    def __init__(self, args):
        super(ErrJudgePromptDataset, self).__init__(args)
        self.error_correct = args.error_correct  #是否在训练judger的时候进行token纠错
        self.prompt = '这个句子"{}"中有错别字吗？{}。'

    def __getitem__(self, item):
        label = self.labels[item]
        is_err = "有" if len(label.strip().split(",")) >= 3 else "无"
        input_text = self.prompt.format(self.texts[item], "_")

        if self.error_correct:
            tokens = list(self.texts[item])
            label_splits = label.split(",")
            for i in range(1, len(label_splits), 2):
                if i + 1 >= len(label_splits):
                    break
                pos = int(label_splits[i]) - 1
                token = label_splits[i + 1].strip()
                tokens[pos] = token
            correct_text = "".join(tokens)
            label_text = self.prompt.format(correct_text, is_err)
        else:
            label_text = self.prompt.format(self.texts[item], is_err)

        sentence_id = self.ids[item]
        label_idx = len(label_text) - 1
        return sentence_id, input_text, label_text, label_idx
