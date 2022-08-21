import csv
import time

import torch
import torch.optim
import torch.utils.data

from data.data_entry import get_loader, get_dataset
from data.call_func import _get_tokenizer
from model.model_entry import select_model

from utils.args_util import load_args
from trainers.train_base import TrainerBase
from optimizer import get_optimizer

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('TestCscC')
config_yaml = "test_error_judge.yaml"
# ex.add_source_file("./configs/%s" %config_yaml)
observer_mongo = MongoObserver.create(url='', db_name='db')
ex.observers.append(observer_mongo)

#加载训练参数
args = load_args("./configs/%s"%config_yaml)
# 超参数设置
ex.add_config(args)

@ex.automain
def main():
    train_dataset = get_dataset(args.dataset.train)
    test_dataset = get_dataset(args.dataset.test)
    train_loader = get_loader(train_dataset, args.loader.train)
    test_loader = get_loader(test_dataset, args.loader.test)

    args.train_paras.cuda = "cuda:{}".format(args.train_paras.cuda_id) if torch.cuda.is_available() else "cpu"
    print(args.train_paras.cuda)
    model = select_model(args.model).to(args.train_paras.cuda)

    optimizer = get_optimizer(model, args.optimizer)

    trainer = TrainerBase(optimizer, train_loader, test_loader, model, args.train_paras, ex, _get_tokenizer())
    trainer.train()


if __name__ == '__main__':
    main()































