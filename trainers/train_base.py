import os

import numpy as np
import torch
from dotted_dict import DottedDict
from sklearn import  metrics
from os.path import join
from glob import glob
import re
from torch.optim import lr_scheduler
from torchcontrib.optim import SWA


class TrainerBase:
    def __init__(self, optimizer, train_loader, val_loader, model, args, sacred_ex, tokenizer):
        self.args = args
        self.cuda = args.cuda
        self.tokenizer = tokenizer


        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        try:
            self.model.module.encoder.resize_token_embeddings(len(self.tokenizer))
        except:
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        self.optimizer = optimizer
        self.metric = torch.nn.CrossEntropyLoss()

        self.steps_cnt = 0
        self.epochs_cnt = 0


        self.ex = sacred_ex

        self.best_f1 = 0


    def train(self):
        for epoch in range(self.args.epochs):
            self.epochs_cnt = epoch+1
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            # self.logger.save_curves(epoch)
            # self.logger.save_check_point(self.model, epoch)

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        predict_logits = []
        true_labels = []



        for i, data in enumerate(self.train_loader):
            inputs = self.step(data)
            # torch.save(inputs,"./tests/inputs.pkl")
            predicts = self.model(inputs)

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss = self.compute_loss(predicts, inputs['judge_labels'])

            loss.backward()
            self.optimizer.step()

            predict_logits.append(predicts.detach().cpu().numpy())
            true_labels.append(inputs['judge_labels'].detach().cpu().numpy())
            self.ex.log_scalar("train_batch_loss", loss.item())
            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))


        predict_logits = np.concatenate(predict_logits, axis=0)
        predict_labels = predict_logits.argmax(axis=1)
        true_labels = np.concatenate(true_labels).reshape(-1).tolist()
        metrics = self.compute_metrics(predict_labels, true_labels, prefix="train")
        for k, v in metrics.items():
            print(k)
            print(v)

        # self.scheduler.step()


    def val_per_epoch(self, epoch):
        self.optimizer.swap_swa_sgd()
        self.model.eval()
        predict_logits = []
        true_labels = []
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs = self.step(data)
                predicts = self.model(inputs)
                predict_logits.append(predicts.cpu().numpy())
                true_labels.append(inputs['judge_labels'].cpu().numpy())
        predict_logits = np.concatenate(predict_logits, axis=0)
        predict_labels = predict_logits.argmax(axis=1).reshape(-1).tolist()
        true_labels = np.concatenate(true_labels).reshape(-1).tolist()
        metrics = self.compute_metrics(predict_labels, true_labels, prefix="val")
        if metrics.f1 > self.best_f1:
            self.best_f1 = metrics.f1
            self.save_model(self.args.save_dir, self.args.max_model_num)
        for k, v in metrics.items():
            print(k)
            print(v)
        self.optimizer.swap_swa_sgd()




    def step(self, data):
        self.steps_cnt += 1
        inputs = data
        for k in inputs.keys():
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(self.cuda)

        return inputs


    def compute_metrics(self, pred, gt , prefix = ""):
        # you can call functions in metrics.py
        metric_result = DottedDict()
        metric_result.accuracy = metrics.accuracy_score(gt, pred)
        metric_result.precision = metrics.precision_score(gt, pred)
        metric_result.recall = metrics.recall_score(gt, pred)
        metric_result.f1 = metrics.f1_score(gt, pred)
        metric_result.report = metrics.classification_report(gt, pred)

        for k in metric_result.keys():
            if isinstance(metric_result[k],float):
                self.ex.log_scalar("%s_%s" % (prefix, k), metric_result[k])

        return metric_result


    def compute_loss(self, pred, gt):
        if self.args.loss == 'l1':
            loss = (pred - gt).abs().mean()
        elif self.args.loss == 'ce':
            loss = self.metric(pred, gt)
        else:
            loss = torch.nn.functional.mse_loss(pred, gt)
        return loss

    def save_model(self, save_dir, max_model_num=10):
        name = "model_epoch%s_f1%s.pkl" % (self.epochs_cnt, self.best_f1)
        model = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {"model_state_dict": model.state_dict(),
                      # "optimizer_state_dic": self.optimizer.state_dict(),
                      "epoch": self.epochs_cnt,
                      "tokenizer": self.tokenizer}

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(checkpoint, join(save_dir, name))
        saved_models = list(glob(join(save_dir, "model_epoch*.ckpt")))
        if len(saved_models) > max_model_num:
            epochs = [int(re.findall("epoch(\d+)_", p)[0]) for p in saved_models]
            min_f1_model = saved_models[epochs.index(min(epochs))]
            os.remove(min_f1_model)















