# coding=utf-8
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pypinyin
import torch
from collections import defaultdict
from utils import word2pinyin,word2pinyin_similar
import random
from pytorch_transformers import BertTokenizer
from utils import get_error_type

def load_sighan(input_file,label_file):
    with open(input_file,"r",encoding="utf-8") as f:
        lines = f.readlines()
    ids,texts = [],[]
    for l in lines:
        ids.append(l[0:l.index(")")+1])
        texts.append(l[l.index(")")+1:].strip())
    with open(label_file,"r",encoding="utf-8") as f:
        lines = f.readlines()
    label_ids,infos = [],[]
    for l in lines:
        splits = [s.strip() for s in l.split(",")]

        label_ids.append(splits[0])
        info = []
        for i in range(1,len(splits),2):
            info.append(splits[i:i+2])
        infos.append(info)
    df1 = pd.DataFrame({"id":ids,'text':texts})
    df2 = pd.DataFrame({"id":label_ids,"label":infos})
    df = pd.merge(left=df1,right=df2,how="inner")
    # assert len(df1) == len(df2) and len(df2) == len(df)
    print(len(df1),len(df2),len(df))
    return df.id,df.text,df.label



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

class SigDataset(Dataset):
    def __init__(self, ids, texts,labels):
        self.ids = ids
        self.labels = labels
        self.texts = texts


    def __getitem__(self, item):
        text = self.texts[item].strip().lower()
        for label in self.labels[item]:
            if len(label) ==0 or (not is_number(label[0])):
                text = "这个句子没有错误"
                self.labels[item] = {('0',)}
                continue
            if int(label[0]) > len(text):
                text = "这个句子没有错误"
                self.labels[item] = {('0',)}
        return text, self.ids[item],list(self.labels[item])

    def __len__(self):
        return len(self.ids)

def unk():
    return '[UNK]'

class SigLoaderBuilder:
    def __init__(self, tokenizer0,tokenizer,args):
        self.tokenizer0 = tokenizer0
        self.tokenizer = tokenizer
        self.word_num = len(tokenizer)
        self.max_len = args.max_len
        if args.pinyin_type == "same":
            self.word2pinyinDict = {k:word2pinyin(k) for k in [tokenizer.convert_ids_to_tokens([i])[0] for i in range(len(tokenizer))]}
            self.word2pinyinDict = defaultdict(unk,self.word2pinyinDict)
        elif args.pinyin_type == "similar":
            assert 1==2
            self.word2pinyinDict = {k: word2pinyin_similar(k) for k in tokenizer.vocab.keys()}
            self.word2pinyinDict = defaultdict(unk, self.word2pinyinDict)
        self.args = args
        self.pinyin=False


    def textes2ids(self, tokens):
        # encoded = self.tokenizer.batch_encode_plus(texts, max_length=self.max_len, pad_to_max_length='right')
        input_ids,pinyin_ids = self.encode_texts(tokens)
        return input_ids,pinyin_ids

    def shape_similarity(self,pair):
        w, w1 = pair
        id, id1 = self.tokenizer.convert_tokens_to_ids([w, w1])
        e, e1 = self.shape_embeds(torch.LongTensor([[id], [id1]]))
        similarity = torch.cosine_similarity(e, e1)
        return similarity.item()

    def is_same_pinyin(self,pair):
        w, w1 = pair
        return word2pinyin(w) == word2pinyin(w1)

    def pair2type(self,pair):
        typo = "语义"
        pinyin_same = self.is_same_pinyin(pair)
        shape_similar =self.shape_similarity(pair) >= 0.7
        if pinyin_same and not shape_similar:
            typo = "字音"
        elif pinyin_same and shape_similar:
            typo = "字音字形"
        elif not pinyin_same and shape_similar:
            typo = "字形"
        return typo

    def pair2type_id(self,pair):
        if pair[0] == pair[1]:
            return 0
        typo = 1
        pinyin_same = self.is_same_pinyin(pair)
        shape_similar = self.shape_similarity(pair) >= 0.7
        if pinyin_same and not shape_similar:
            typo = 2
        elif pinyin_same and shape_similar:
            typo = 3
        elif not pinyin_same and shape_similar:
            typo = 4
        return typo

    def replace_right_token(self,tokens,labels):

        new_tokens = [w for  w in tokens]
        mask_labels = np.ones(max(len(new_tokens),self.max_len))
        if len(labels[0]) > 1:
            for pos, right_token in labels:
                pos = int(pos)-1
                new_tokens[pos] = right_token
                tokens[pos] = '[MASK]'
        # print("masked ids:",mask_labels)
        return new_tokens,mask_labels[:self.max_len]

    def replace_right_token_no_mask(self,tokens,labels):

        new_tokens = [w for  w in tokens]
        mask_labels = np.zeros(max(len(new_tokens),self.max_len))
        # print("this ",tokens,labels)
        if len(labels) > 0:
            if len(labels[0]) > 1:
                for pos, right_token in labels:
                    pos = int(pos)-1
                    new_tokens[pos] = right_token
                    if pos+1 < self.max_len:
                        mask_labels[pos+1] = 1
        errors = np.zeros((max(len(new_tokens),self.max_len),3))
        for pos,(token,new_token) in enumerate(zip(tokens,new_tokens)):
            errors[pos] = get_error_type(token,new_token)

        return new_tokens,errors[:self.max_len]


    def no_mask_call_func(self, batch_data):
        # print(batch_data)
        texts, ids,labels_list = zip(*batch_data)
        tokens_list = [ list(text)   for text in texts]
        self.max_len = max([len(l) for l in tokens_list])
        self.max_len = min(self.max_len,200)
        # print(texts)
        _,pinyin_ids = self.textes2ids(tokens_list)#处理后的bert的输入

        #准备中间数据的标签
        new_tokens_list,mask_labels_list = zip(*[self.replace_right_token_no_mask(tokens,labels) for tokens,labels in zip(tokens_list,labels_list)])
        mask_labels_list = np.stack(mask_labels_list)
        input_ids,_ = self.textes2ids(tokens_list)#处理后的bert的输入
        labels,pinyin_ids_label = self.textes2ids(new_tokens_list)#原本的输入对应标签和拼音标签
        # print("errors:",mask_labels_list[:3])
        # print(mask_labels_list.shape)
        # print(encodes['input_ids'][0])
        # print(texts[0])
        # print(new_texts[0])
        # print(mask_labels_list[0])
        if self.pinyin:
            return input_ids,pinyin_ids,labels,mask_labels_list,pinyin_ids_label
        if self.err_type=='err_type':
            print("use error type")
            error_type_list = []
            for i in range(len(tokens_list)):
                errs = []
                for j in range(len(tokens_list[i])):
                    errs.append(self.pair2type_id((tokens_list[i][j],new_tokens_list[i][j])))
                error_type_list.append(errs)
            error_type_list = self.pad_ids(error_type_list)
            return input_ids, pinyin_ids, labels, error_type_list
        return input_ids, pinyin_ids, labels, mask_labels_list
    def call_random_mask_func(self,batch_data):
        texts, ids, labels_list = zip(*batch_data)
        tokens_list = [list(text) for text in texts]
        self.max_len = max([len(l) for l in tokens_list])

        input_ids, pinyin_ids = self.textes2ids(tokens_list)  # 处理后的bert的输入

        new_tokens_list, mask_labels_list = zip(*[self.replace_random_tokens(tokens) for tokens, labels in zip(tokens_list, labels_list)])

        mask_labels_list = np.stack(mask_labels_list)
        # , _ = self.textes2ids(tokens_list)  # 处理后的bert的输入#之前mask掉了，但是现在并没有
        labels, _ = self.textes2ids(new_tokens_list)  # 原本的输入对应标签和拼音标签
        # print(mask_labels_list.shape)
        # print(encodes['input_ids'][0])
        # print(texts[0])
        # print(new_texts[0])
        # print(mask_labels_list[0])
        assert "not implemented" == ''
        return input_ids, pinyin_ids, labels, mask_labels_list


    def get_title_loader(self, dataset, batch_size, shuffle):
        print("use no mask!!!")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.no_mask_call_func)
        return loader

    def prepare_test_sentence(self,sentence):
        sentence = sentence.lower().strip()
        tokens = list(sentence)
        input_ids,pinyin_ids = self.textes2ids([tokens])
        return input_ids,pinyin_ids



    def get_distributed_loader(self,dataset, batch_size,train_sampler, kwargs):
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=self.call_func, **kwargs)
        return train_loader

    def pad_ids(self,ids_list):
        max_len = self.max_len
        rs_list = []
        for ids in ids_list:
            if len(ids) < max_len - 2:
                ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id] * (
                            max_len - 2 - len(ids))
            elif len(ids) == max_len - 2:
                ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
            else:
                ids = [self.tokenizer.cls_token_id] + ids[:max_len - 2] + [self.tokenizer.sep_token_id]
            rs_list.append(ids)
        return rs_list

    def encode_texts(self, tokens_list):
        pinyin_ids_list = [[self.word2pinyinDict[token] for token in tokens] for tokens in tokens_list]
        ids_list = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list]
        pinyin_ids_list = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in pinyin_ids_list]
        # print(self.tokenizer.convert_ids_to_tokens(pinyin_ids_list[0]))

        ids_list = self.pad_ids(ids_list)
        pinyin_ids_list = self.pad_ids(pinyin_ids_list)
        return ids_list, pinyin_ids_list




if __name__ == '__main__':
    import os
    os.chdir("../")
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument("--bert_dir", type=str, default="./resources/bert_base_chinese")
    paser.add_argument("--share_weights", type=str, default="none", help="same_bert|none")
    paser.add_argument("--max_len",type=int,default=40)
    paser.add_argument("--pinyin_type", type=str, default="same", help="same|similar")




    args = paser.parse_args()
    assert args.share_weights in ['same_bert','none']


    tokenizer = BertTokenizer.from_pretrained(r"resources/bert_base_chinese")
    print(tokenizer)
    print("tokenizer init:", len(tokenizer))
    from os.path import  join
    pinyins = [(word2pinyin(w) if word2pinyin(w) != w else None) for w in list(tokenizer.vocab.keys())]
    pinyins = list(set(pinyins))
    pinyins.remove(None)
    tokenizer.add_tokens(pinyins)

    tokenizer0 =  BertTokenizer.from_pretrained(r"resources/bert_base_chinese")



    input_file = r"resources/merged_sig/extend_training_input_shuf.txt"
    label_file = r"resources/merged_sig/extend_training_truth_shuf.txt"
    ids, texts, labels = load_sighan(input_file,label_file)
    tds = SigDataset(ids, texts,labels)
    print(ids)

    tdb = SigLoaderBuilder(tokenizer0,tokenizer,args)
    loader = tdb.get_title_loader(tds, 2, True)

    # input_ids,pinyin_ids,labels,mask_labels_list = next(loader.__iter__())
    for input_ids,pinyin_ids,labels,mask_labels_list  in loader:
        print(input_ids,pinyin_ids,labels,mask_labels_list)
        print(11111111111111111111111111111)
        break
    # print(input_ids[0])
    # print(labels[0])
    # print(mask_labels_list[0])
    # new_text,mask_labels_list = tdb.replace_15("思思深情演唱《做你心上的人》，独特的嗓音，别有一番滋味在心头")
    # print(new_text,mask_labels_list)
