import os.path

from .train_base import TrainerBase
import numpy as np
import torch
import pandas as pd
from data.call_func import _get_tokenizer
from dotted_dict import DottedDict
from metrics import compute_csc_metrics_by_file
from torch import nn
from os.path import join
from glob import glob
import re
from trainers.train_pipeline import TrainerPipelineCorrector

class TrainerPipelineCorrectorMulti(TrainerPipelineCorrector):
    def __init__(self, optimizer, train_loader, val_loader, model, args, sacred_ex):
        super(TrainerPipelineCorrectorMulti, self).__init__(optimizer, train_loader, val_loader, model, args, sacred_ex)

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        predict_labels = []
        input_ids = []
        sentence_ids = []
        for i, data in enumerate(self.train_loader):
            inputs = self.step(data)
            select_logits, detector_encoded, selector_encoded, select_labels, correct_labels1 = self.model(inputs)
            # compute gradient and do Adam step
            token_logits = detector_encoded['logits']
            loss1 = self.compute_loss(token_logits.reshape(-1, token_logits.shape[-1]), inputs['correct_labels'].reshape(-1))
            token_logits1 = selector_encoded['logits']
            loss2 = self.compute_loss(token_logits1.reshape(-1, token_logits1.shape[-1]), correct_labels1.reshape(-1))
            # loss2 = self.compute_loss(select_logits.reshape(-1, select_logits.shape[-1]), select_labels.reshape(-1))
            loss = loss1 + loss2
            loss1.backward()


            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.zero_grad()

            predict_labels.extend(token_logits1.argmax(dim=2).detach().cpu().numpy().tolist())
            input_ids.extend(inputs['input_ids'].detach().cpu().numpy().tolist())
            sentence_ids.extend(inputs['sentence_ids'])
            self.ex.log_scalar("train_batch_loss1", loss1.item())
            self.ex.log_scalar("train_batch_loss2", loss2.item())
            self.ex.log_scalar("train_batch_loss", loss.item())
            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
                # print(inputs.keys())
                # print(inputs['sentence_ids'][0])
                # print("input_ids:", "".join(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].detach().cpu().tolist())))
                # print("pinyin_ids:", " ".join(self.tokenizer.convert_ids_to_tokens(inputs['pinyin_ids'][0].detach().cpu().tolist())))
                # print("position_ids:", inputs['position_ids'][0].detach().cpu().tolist())
                # print("token_type_ids:", inputs['token_type_ids'][0].detach().cpu().tolist())




        # predict_labels = np.concatenate(predict_labels, axis=0)
        # true_labels = np.concatenate(true_labels, axis= 0 )

        save_path = os.path.join(self.args.save_dir, "train_predicts_%s" % str(self.epochs_cnt))
        self.write_csc_predictions(input_ids, predict_labels, sentence_ids, save_path)
        metrics = self.compute_csc_metrics_by_file(save_path, self.args.train_label_file, show=True)
        self.log_metrics(metrics, prefix="train")
        # self.scheduler.step()


    def val_per_epoch(self, epoch):
        if self.args.use_swa:
            self.optimizer.swap_swa_sgd()
        self.model.eval()
        predict_labels = []
        input_ids = []
        sentence_ids = []
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs = self.step(data)
                select_logits, detector_encoded, selector_encoded, select_labels, correct_labels1 = self.model(inputs)
                # compute gradient and do Adam step
                token_logits = selector_encoded['logits']

                predict_labels.extend(token_logits.argmax(dim=2).detach().cpu().numpy().tolist())
                input_ids.extend(inputs['input_ids'].detach().cpu().numpy().tolist())
                sentence_ids.extend(inputs['sentence_ids'])

        save_path = os.path.join(self.args.save_dir, "val_predicts_%s" % str(self.epochs_cnt))
        self.write_csc_predictions(input_ids, predict_labels, sentence_ids, save_path)
        metrics = self.compute_csc_metrics_by_file(save_path, self.args.val_label_file, show=True)
        self.log_metrics(metrics, prefix="val")

        if metrics.f1 > self.best_f1:
            self.best_f1 = metrics.f1
            self.save_model(self.args.save_dir, self.args.max_model_num)

        for k, v in metrics.items():
            print(k)
            print(v)
        if self.args.use_swa:
            self.optimizer.swap_swa_sgd()










