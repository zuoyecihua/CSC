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

class TrainerPipelineCorrector:
    def __init__(self, optimizer, train_loader, val_loader, model, args, sacred_ex):
        self.args = args
        self.ex = sacred_ex
        self.tokenizer = _get_tokenizer()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model.detector.resize_token_embeddings(len(self.tokenizer))
        self.model.selector.resize_token_embeddings(len(self.tokenizer))
        self.steps_cnt = 0
        self.cuda = self.args.cuda_id
        self.best_f1 = 0

        self.crossEntropy = torch.nn.CrossEntropyLoss()

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        predict_labels = []
        input_ids = []
        sentence_ids = []
        for i, data in enumerate(self.train_loader):
            inputs = self.step(data)
            select_logits, detector_encoded, selector_encoded, select_labels, correct_labels = self.model(inputs)
            # compute gradient and do Adam step
            token_logits = detector_encoded['logits']
            loss1 = self.compute_loss(token_logits.reshape(-1, token_logits.shape[-1]), inputs['correct_labels'].reshape(-1))
            loss2 = self.compute_loss(select_logits.reshape(-1, select_logits.shape[-1]), select_labels.reshape(-1))
            loss = loss1 + loss2
            loss1.backward()


            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.zero_grad()

            predict_labels.extend(token_logits.argmax(dim=2).detach().cpu().numpy().tolist())
            input_ids.extend(inputs['input_ids'].detach().cpu().numpy().tolist())
            sentence_ids.extend(inputs['sentence_ids'])
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

    def log_metrics(self, metrics, prefix= "train"):

        flattened_metrics = pd.json_normalize(metrics, sep="_").to_dict(orient='records')[0]
        for k in flattened_metrics.keys():
            print(k, flattened_metrics[k])
            self.ex.log_scalar("%s_%s" % (prefix, k), flattened_metrics[k])


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
                select_logits, detector_encoded, selector_encoded, select_labels, correct_labels = self.model(inputs)
                # compute gradient and do Adam step
                token_logits = detector_encoded['logits']

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

    def compute_csc_metrics_by_file(self, predict_file, labels_file, show = False):
        return compute_csc_metrics_by_file(predict_file, labels_file, show)




    def write_csc_predictions(self, input_ids, predict_labels, sentence_ids, save_path):
        tokenizer = _get_tokenizer()
        comma_token_id = tokenizer.convert_tokens_to_ids([","])[0]
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, "w", encoding="utf-8") as f:
            # print(len(input_ids),len(predict_labels),len(sentence_ids))
            punctuations = ['"', '”', ">", "》", "]", "】", "\n"]
            for tl,pl,sid in zip(input_ids,predict_labels,sentence_ids):
                # print(sid)
                # print("true_labels:",tokenizer.convert_ids_to_tokens(tl))
                # print("predict_labels:", tokenizer.convert_ids_to_tokens(pl))
                corrected = [sid]
                for i in range(len(tl)):
                    if tl[i] == tokenizer.sep_token_id:
                        break
                    if tl[i] != pl[i] and pl[i] != comma_token_id:
                        pred_token = tokenizer.convert_ids_to_tokens([pl[i]])[0]
                        if len(pred_token) == 1:
                            token = tokenizer.convert_ids_to_tokens([pl[i]])[0]
                            if token not in punctuations and len(token.strip()) > 0:
                                corrected.extend([i, token])
                if len(corrected) == 0:
                    corrected.append(0)
                corrected_str = ", ".join([str(c) for c in corrected]) + "\n"
                f.write(corrected_str)

    def process_detect_logits_for_single_item(self, raw_item , detec_logits_probablity):
        detec_predicts = detec_logits_probablity.argmax(dim=-1)
        detec_tokens = self.tokenizer.convert_ids_to_tokens(detec_predicts)[1:-1]
        id, text, label = raw_item
        tokens = list(text)
        positions = list(range(len(tokens) + 1))
        label_splits = label.split(",")
        all_candidates = []
        for i in range(len(text) - 1):
            text = text.lower()
            j = 0
            minus = j if i > 0 else 0
            added = j if i < len(text) - 1 else 0
            logit = detec_logits_probablity[i + 1][detec_predicts[i + 1]].item()

            if text[i] != detec_tokens[i] or text[i - minus] != detec_tokens[i - minus] or text[i + added] != \
                    detec_tokens[i + added] or logit < 0.8:
                if detec_tokens[i] == '，':
                    continue
                pos = i
                origin_token = tokens[pos]
                candidates = self.word2candidates[origin_token]
                cids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(candidates))
                clogits = detec_logits_probablity[pos + 1][cids].tolist()

                good = [x[1] for x in sorted(list(zip(clogits, candidates)), key=lambda x: -x[0])[:5]]
                candidates = list(set("".join(good) + origin_token + detec_tokens[i]))
                all_candidates += candidates
                positions += [pos + 1] * len("".join(candidates))  # tokenizer会增加cls，所以pos + 1
        input_text = ["".join(tokens), "".join(all_candidates)]

        tokens = list(text)
        for i in range(1, len(label_splits), 2):
            if i + 1 >= len(label_splits):
                break
            pos = int(label_splits[i]) - 1
            token = label_splits[i + 1].strip()
            tokens[pos] = token
        label_text = ["".join(tokens), "".join(all_candidates)]

        positions += [len(positions)]  # for sep
        modified_item = (id, input_text, positions, label_text)
        return modified_item

    def compute_loss(self, pred, gt):
        if self.args.loss == 'l1':
            loss = (pred - gt).abs().mean()
        elif self.args.loss == 'ce':
            loss = self.crossEntropy(pred, gt)
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

    def train(self):
        print(self.args)
        for epoch in range(self.args.epochs):
            self.epochs_cnt = epoch+1
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            # self.logger.save_curves(epoch)
            # self.logger.save_check_point(self.model, epoch)

    def step(self, data):
        self.steps_cnt += 1
        inputs = data
        for k in inputs.keys():
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(self.cuda)

        return inputs









