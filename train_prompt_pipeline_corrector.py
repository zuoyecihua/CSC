# -*- coding: UTF-8 -*-

import csv
import time

import torch
import torch.optim
import torch.utils.data

from data.data_entry import get_loader, get_dataset
from model.model_entry import select_model

from utils.args_util import load_args
from trainers.train_pipeline import TrainerPipelineCorrector
from optimizer import get_optimizer

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('PromptorPipeline')
config_yaml = "test_csc_pipeline.yaml"
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

    if args.load_model_path != "":
        print("loding model from ", args.load_model_path)
        model = torch.load(args.load_model_path, map_location=torch.device("cpu")).to(args.train_paras.cuda)
    else:
        model = select_model(args.model).to(args.train_paras.cuda)

    optimizer = get_optimizer(model, args.optimizer)

    trainer = TrainerPipelineCorrector(optimizer, train_loader, test_loader, model, args.train_paras, sacred_ex = ex)
    trainer.train()


if __name__ == '__main__':
    main()































