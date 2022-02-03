import os.path

from .train_base import TrainerBase
import numpy as np
import torch
import pandas as pd
from data.call_func import _get_tokenizer
from dotted_dict import DottedDict
from metrics import compute_csc_metrics_by_file

class TrainerPromptJudger(TrainerBase):
    def __init__(self, optimizer, train_loader, val_loader, model, args, sacred_ex):
        self.tokenizer = _get_tokenizer()
        super(TrainerPromptJudger, self).__init__(optimizer, train_loader, val_loader, model, args, sacred_ex, self.tokenizer)

        self.model.module.encoder.resize_token_embeddings(len(self.tokenizer))

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        predict_labels = []
        true_labels = []
        sentence_ids = []
        for i, data in enumerate(self.train_loader):
            inputs = self.step(data)
            token_logits = self.model(inputs)
            # compute gradient and do Adam step
            #prompt 有三种损失函数： 1. 只计算prompt 2. 计算所有token
            # print(inputs['label_ids'].shape,inputs['answer_idxes'].shape)
            prompt_labels = torch.gather(inputs['label_ids'], 1, inputs['answer_idxes']).reshape(-1)
            inputs['answer_idxes'] = inputs['answer_idxes'].unsqueeze(2).expand( (inputs['answer_idxes'].shape[0], 1, token_logits.shape[-1]))

            # print(token_logits.shape, inputs['answer_idxes'].shape)
            prompt_logits = torch.gather(token_logits, 1, inputs['answer_idxes'])
            prompt_logits = prompt_logits.reshape(-1, prompt_logits.shape[-1])

            # print(prompt_logits.shape, prompt_labels.shape)
            loss_prompt = self.compute_loss(prompt_logits , prompt_labels.reshape(-1))
            self.ex.log_scalar("train_loss_prompt", loss_prompt.item())
            if self.args.token_loss:
                loss_token = self.compute_loss(token_logits.reshape(-1, token_logits.shape[-1]), inputs['label_ids'].reshape(-1))
                self.ex.log_scalar("train_loss_token", loss_token.item())
                loss = loss_prompt + loss_token
            else:
                loss = loss_prompt

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.module.step()
            self.model.module.zero_grad()

            predict_labels.append(prompt_logits.argmax(dim=1).detach().cpu().numpy())
            true_labels.append(prompt_labels.detach().cpu().numpy())
            sentence_ids.extend(inputs['sentence_ids'])

            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))
                print(prompt_logits.shape)
                print("input_ids:", "".join(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].detach().cpu().numpy().tolist())))
                print("label_ids:","".join(self.tokenizer.convert_ids_to_tokens(inputs['label_ids'][0].detach().cpu().numpy().tolist())))
                print("token_logits:","".join(self.tokenizer.convert_ids_to_tokens(token_logits[0].argmax(dim=1).detach().cpu().numpy().tolist())))
                print("prompt_logits:","".join(self.tokenizer.convert_ids_to_tokens(prompt_logits.argmax(dim=1).reshape(-1).detach().cpu().numpy().tolist())))
                print("prompt_labels:","".join(self.tokenizer.convert_ids_to_tokens(
                    prompt_labels.reshape(-1).detach().cpu().numpy().tolist())))
                # torch.save((inputs,token_logits,prompt_logits),"./tests/tmp.pkl")


        predict_labels = np.concatenate(predict_labels, axis=0)
        true_labels = np.concatenate(true_labels).reshape(-1).tolist()
        metrics = self.compute_metrics(predict_labels, true_labels, prefix="train")
        for k, v in metrics.items():
            print(k)
            print(v)
        self.log_metrics(metrics, prefix="train")
        # self.scheduler.step()

    def log_metrics(self, metrics, prefix= "train"):

        flattened_metrics = pd.json_normalize(metrics, sep="_").to_dict(orient='records')[0]
        for k in flattened_metrics.keys():
            print(k, flattened_metrics[k])
            self.ex.log_scalar("%s_%s" % (prefix, k), flattened_metrics[k])


    def val_per_epoch(self, epoch):
        if self.args.use_swa:
            self.optimizer.module.swap_swa_sgd()
        self.model.eval()
        predict_labels = []
        true_labels = []
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs = self.step(data)
                token_logits = self.model(inputs)
                prompt_labels = torch.gather(inputs['label_ids'], 1, inputs['answer_idxes']).reshape(-1)
                inputs['answer_idxes'] = inputs['answer_idxes'].unsqueeze(2).expand(
                    (inputs['answer_idxes'].shape[0], 1, token_logits.shape[-1]))
                # print(token_logits.shape, inputs['answer_idxes'].shape)
                prompt_logits = torch.gather(token_logits, 1, inputs['answer_idxes'])

                predict_labels.append(prompt_logits.argmax(dim=2).cpu().numpy())
                true_labels.append(prompt_labels.cpu().numpy())
        predict_labels = np.concatenate(predict_labels, axis=0).reshape(-1).tolist()
        true_labels = np.concatenate(true_labels).reshape(-1).tolist()
        metrics = self.compute_metrics(predict_labels, true_labels, prefix="val")

        if metrics.f1 > self.best_f1:
            self.best_f1 = metrics.f1
            self.save_model(self.args.save_dir, self.args.max_model_num)
        for k, v in metrics.items():
            print(k)
            print(v)
        if self.args.use_swa:
            self.optimizer.module.swap_swa_sgd()

    def compute_metrics(self,pred, gt , prefix="val"):
        flage_err = self.tokenizer.convert_tokens_to_ids(['有'])[0]
        gt = [(1 if x==flage_err else 0) for x in gt ]
        pred = [(1 if x == flage_err else 0) for x in pred]
        return super(TrainerPromptJudger, self).compute_metrics(pred, gt, prefix)

    def step(self, data):
        self.steps_cnt += 1
        inputs = data
        for k in inputs.keys():
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].cuda()

        return inputs



