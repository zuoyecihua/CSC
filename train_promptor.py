import csv
import time

import torch
import torch.optim
import torch.utils.data

from data.data_entry import get_loader, get_dataset
from data.call_func import _get_tokenizer
from model.model_entry import select_model

from utils.args_util import load_args
from trainers.train_promptor import TrainerPromptJudger
from optimizer import get_optimizer

from sacred import Experiment
from sacred.observers import MongoObserver
import os




ex = Experiment('TestPromptor')
config_yaml = "test_promptor_judge.yaml"
# ex.add_source_file("./configs/%s" %config_yaml)
observer_mongo = MongoObserver.create(url='mongodb://admin:123456@114.212.86.198:27017/?authMechanism=SCRAM-SHA-1', db_name='db')
ex.observers.append(observer_mongo)

#加载训练参数
args = load_args("./configs/%s"%config_yaml)
# 超参数设置
ex.add_config(args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args.train_paras.cuda_id = [int(x.strip()) for x in args.train_paras.cuda_id.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in args.train_paras.cuda_id])
args.train_paras.cuda_id = [i for i,_ in enumerate(args.train_paras.cuda_id)]
print(torch.cuda.is_available(), torch.cuda.current_device())
@ex.automain
def main():
    train_dataset = get_dataset(args.dataset.train)
    test_dataset = get_dataset(args.dataset.test)
    train_loader = get_loader(train_dataset, args.loader.train)
    test_loader = get_loader(test_dataset, args.loader.test)

    args.train_paras.cuda = "cuda:{}".format(args.train_paras.cuda_id) if torch.cuda.is_available() else "cpu"
    print(args.train_paras.cuda)
    print(args.train_paras.cuda_id)
    model = torch.nn.DataParallel (select_model(args.model), device_ids=args.train_paras.cuda_id).cuda()

    optimizer = get_optimizer(model, args.optimizer)
    optimizer = torch.nn.DataParallel(optimizer, device_ids=args.train_paras.cuda_id)

    trainer = TrainerPromptJudger(optimizer, train_loader, test_loader, model, args.train_paras, ex)
    trainer.train()


if __name__ == '__main__':
    main()































