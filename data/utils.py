import pypinyin
from collections import defaultdict

def word2pinyin(w):
    return pypinyin.lazy_pinyin(list(w))

def word2pinyin_similar(word):
    # print(word)
    pinyins = pypinyin.pinyin(word, style=pypinyin.NORMAL)
    if len(pinyins) > 0:
        pinyin = "".join(pinyins[0])
    else:
        pinyin = ""
    if len(word) == 1 and len(pinyin) >= 1:
        pinyin = pinyin.replace("zh","z").replace("ch","c").replace("sh","s")\
            .replace("ang","an").replace("eng","en").replace("ing","in").replace("ong","on")\
            .replace("v","u")#平舌音：baiz、c、dus。翘舌音：zh、ch、zhish、r。前鼻音：an、en、in、un、ün。后鼻音：ang、eng、ing、ong。
    return pinyin

def text2pinyin_similar(text):
    return [word2pinyin_similar(w) for w in text]


def build_candidates_dict():
    #取得一个汉字的近形字和近音字
    word2candidates = defaultdict(set)
    with open(r"resources/gcn_graph.ty_xj/SimilarPronunciation_simplied.txt","r",encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for l in  lines:
            w,candiates = [x.strip() for x in l.split("\t",1)]
            if w =='户':
                print(candiates)
            word2candidates[w].update(candiates)
    with open(r"resources/gcn_graph.ty_xj/SimilarShape_simplied.txt","r",encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for l in  lines:
            w,candiates = [x.strip() for x in l.split(",",1)]
            word2candidates[w].update(candiates)
    return word2candidates

def add_candidates():
    text_file, label_file = "./resources/merged/clean_extend_training_input_shuf_v1.txt","./resources/merged/clean_extend_training_truth_shuf_v1.txt"
    w2c = defaultdict(set)
    f1,f2 = open(text_file,"r",encoding="utf-8"),open(label_file,"r",encoding="utf-8")
    for text,label in zip(list(f1),list(f2)):
        text = text.split("\t")[1].strip()
        label_splits = label.strip().split(",")
        if label_splits[-1] == "":
            del label_splits[-1]
        if len(label_splits) < 3:
            continue
        for i in range(1,len(label_splits),2):
            pos,candidate = label_splits[i],label_splits[i+1]
            idx = int(pos) - 1
            w2c[text[idx]].add(candidate.strip())
    return w2c

def build_candidates_for_all():
    w2c1 = build_candidates_dict()
    wc2c2 = add_candidates()
    w2c = defaultdict(set)
    for k in set(list(w2c1.keys()) + list(wc2c2.keys())):
        w2c[k] = set(list(w2c1[k]) + list(wc2c2[k]))
    return w2c

def pad_positions(seqs,max_len):
    padded = []
    max_len = min(max_len,max([len(seq) for seq in seqs]))
    for seq in seqs:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            seq += list(range(len(seq), max_len))
            padded.append(seq)
    return padded

def pad_0(seqs, max_len):
    padded = []
    max_len = min(max_len, max([len(seq) for seq in seqs]))
    for seq in seqs:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            seq += [0]*(max_len - len(seq))
            padded.append(seq)
    return padded