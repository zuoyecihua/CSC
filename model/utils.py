import torch
from torch import nn
from transformers import BertTokenizer, AlbertModel, AutoModel, BertModel, BertForMaskedLM

_pretrained2loader = {
    "clue/roberta_chinese_base": BertModel,
    "clue/albert_chinese_small": AlbertModel,
    "bert-base-chinese": BertModel
}

_pretrained2Lm = {
    "bert-base-chinese": BertForMaskedLM
}