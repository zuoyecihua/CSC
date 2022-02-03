import os.path

from .train_base import TrainerBase
import numpy as np
import torch
import pandas as pd
from data.call_func import _get_tokenizer
from dotted_dict import DottedDict
from metrics import compute_csc_metrics_by_file
from .utils import ContrastiveCrossEntropyLoss
class TrainerCorrector(TrainerBase):
    def __init__(self, optimizer, train_loader, val_loader, model, args, sacred_ex):
        self.tokenizer = _get_tokenizer()
        super(TrainerCorrector, self).__init__(optimizer, train_loader, val_loader, model, args, sacred_ex, self.tokenizer)

        self.model.encoder.resize_token_embeddings(len(self.tokenizer))

        #准备对比学习损失
        if ContrastiveCrossEntropyLoss in args:
            if args.ContrastiveCrossEntropyLoss:
                print("use contrastive learning!!!")
                self.metric = ContrastiveCrossEntropyLoss(contastive_num=args.contastive_num)



    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        predict_labels = []
        input_ids = []
        sentence_ids = []
        for i, data in enumerate(self.train_loader):
            inputs = self.step(data)
            token_logits = self.model(inputs)
            # compute gradient and do Adam step

            loss = self.compute_loss(token_logits.reshape(-1, token_logits.shape[-1]), inputs['correct_labels'].reshape(-1))
            loss.backward()
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
                token_logits = self.model(inputs)
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

    # def write_csc_predictions(self, input_ids, predict_labels, sentence_ids, save_path):
    #     tokenizer = _get_tokenizer()
    #     from data.utils import build_candidates_dict,add_candidates,word2pinyin_similar
    #     word2candidates = build_candidates_dict()
    #     add_w2c = add_candidates()
    #     comma_token_id = tokenizer.convert_tokens_to_ids([","])[0]
    #     if not os.path.exists(os.path.dirname(save_path)):
    #         os.makedirs(os.path.dirname(save_path))
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         # print(len(input_ids),len(predict_labels),len(sentence_ids))
    #         punctuations = ['"', '”', ">", "》", "]", "】"]
    #         for tl, pl, sid in zip(input_ids, predict_labels, sentence_ids):
    #             corrected = [sid]
    #             for i in range(len(tl)):
    #                 if tl[i] == tokenizer.sep_token_id:
    #                     break
    #                 if tl[i] == pl[i]:
    #                     continue
    #                 pred_token = tokenizer.convert_ids_to_tokens([pl[i]])[0].strip()
    #                 if len(pred_token) != 1 or pred_token in punctuations:
    #                     continue
    #                 true_token = tokenizer.convert_ids_to_tokens(tl[i])[0].strip()
    #                 if pred_token not in word2candidates[true_token] and true_token not in word2candidates[
    #                     pred_token] and true_token not in add_w2c[pred_token] and pred_token not in add_w2c[
    #                     true_token] and word2pinyin_similar(true_token) != word2pinyin_similar(pred_token):
    #                     print(pred_token, true_token)
    #                     continue
    #                     pass
    #                 #                 if pred_token in "得的地":
    #                 #                     continue
    #                 corrected.extend([i, pred_token])
    #             if len(corrected) == 0:
    #                 corrected.append(0)
    #             corrected_str = ", ".join([str(c) for c in corrected]) + "\n"
    #             f.write(corrected_str)





