from torch.utils.data import Dataset
class ErrDetecDataset(Dataset):
    def __init__(self, args):
        text_path, label_path = args.text_path, args.label_path
        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            sentence_ids, texts = zip(*[x.strip().split("\t") for x in lines])
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [x.strip() for x in f.readlines()]
        self.ids = sentence_ids
        self.labels = labels
        # self.texts = texts

        # corrected_texts = []
        # for text,label in zip(texts, labels):
        #     tokens = list(text)
        #     label_splits = label.split(",")
        #     for i in range(1,len(label_splits), 2):
        #         if i + 1 >= len(label_splits):
        #             break
        #         pos = int(label_splits[i]) - 1
        #         tokens[pos] =

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        sentence_id = self.ids

        return sentence_id, text, label

    def __len__(self):
        return len(self.ids)