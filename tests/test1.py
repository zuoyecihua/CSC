import pandas as pd
import sys
import os
from data.data_entry import get_loader, get_dataset
from utils.args_util import load_args

from model.copy_transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
BartForConditionalGeneration().generate()
config_yaml = "test_csc.yaml"
#加载训练参数
args = load_args("./configs/%s"%config_yaml)

train_dataset = get_dataset(args.dataset.train)
test_dataset = get_dataset(args.dataset.test)
train_loader = get_loader(train_dataset, args.loader.train)
test_loader = get_loader(test_dataset, args.loader.test)

if __name__ == '__main__':

    next(train_loader.__iter__())