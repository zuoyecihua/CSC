import torch
from transformers import BertTokenizerFast,BertModel
from data.call_func import _get_tokenizer
from dotted_dict import DottedDict
import torch
import torch.optim
import torch.utils.data
from data.call_func import sighan_error_correct_prompt,sighan_error_detec_csc

from data.data_entry import get_loader, get_dataset
from model.model_entry import select_model

from utils.args_util import load_args
from trainers.train_corrector import TrainerCorrector
from optimizer import get_optimizer
from data.utils import build_candidates_dict
import os
from metrics import compute_csc_metrics_by_file
from model.copy_transformers import AutoModelForMaskedLM, BertForMaskedLM #suport pinyin ids
from utils.random_util import set_random_seed
torch.set_printoptions(precision=2)
import numpy as np
import random
#加载训练参数
seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

config_yaml = "detec_correct_pipeline.yaml"
args = load_args("./configs/%s"%config_yaml)
train_dataset = get_dataset(args.dataset.train)
test_dataset = get_dataset(args.dataset.test)

device = "cuda:1"
def step(data):
    inputs = data
    for k in inputs.keys():
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to(device)
    return inputs


device = "cuda:1"
model = select_model(args.model).to(args.train_paras.cuda_id)
model.encoder.resize_token_embeddings(len(_get_tokenizer()))
model.load_state_dict(torch.load("/home/wangshuai/model",map_location="cpu",errors="strict")) # args.trained_models.detec_model

detec_model = model.to(device)
detec_model.eval()
detec_model.cuda = device
# correct_model = correct_model.to(device)
# correct_model.eval()
# correct_model.cuda = device
detec_model.args.pinyin = True
detec_model.args.position = False
# correct_model.args.pinyin = False
# correct_model.args.position = True


def write_csc_predictions(input_ids, predict_labels, sentence_ids, save_path):
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


from data.data_entry import get_loader, get_dataset
from tqdm import tqdm
val_loader = get_loader(test_dataset, args.loader.test)
model = detec_model
model.eval()
model.encoder.eval()
predict_labels = []
input_ids = []
sentence_ids = []
with torch.no_grad():
    for i, data in tqdm(enumerate(val_loader)):
        inputs = step(data)
        token_logits = model(inputs)
        predict_labels.extend(token_logits.argmax(dim=2).detach().cpu().numpy().tolist())
        input_ids.extend(inputs['input_ids'].detach().cpu().numpy().tolist())
        sentence_ids.extend(inputs['sentence_ids'])

save_path = "tests/test_file"
write_csc_predictions(input_ids, predict_labels, sentence_ids, save_path)
metrics = compute_csc_metrics_by_file(save_path, args.train_paras.val_label_file, show=True)
