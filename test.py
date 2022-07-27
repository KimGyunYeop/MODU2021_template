from calendar import EPOCH
from turtle import forward
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import pandas as pd

from conifg import DATAPATH_DICT
from models import MODEL_DICT
from datasets import DATASET_DICT
from utils import set_seed, torch_MCC, torch_ACC

assert MODEL_DICT.keys() == DATASET_DICT.keys(), "Model and dataset not matched"

#load model path
save_path = "result/tmp"
epoch = 0
setting_json = json.load(open(os.path.join(save_path, 'setting.json'),"r"))

#set config and hyperparameter
task = setting_json["task"]
model_name = setting_json["model_name"]
pretrain_model_path = setting_json["pretrain_model_path"] #korean bert https://huggingface.co/klue/bert-base
save_path = "result/tmp"
gpu = "cuda:0"
device = torch.device(gpu)
batch_size = 128
max_length = 30
EPOCH = 30

#set seed
seed = 42
set_seed(seed)

test_data_path = DATAPATH_DICT[task]["test"]
#load dataset and dataloader
test_dataset = DATASET_DICT[model_name](test_data_path, pretrain_model_path, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

#load model
model = MODEL_DICT[model_name](pretrain_model_path)
model.to(device)

#test phase
all_test_pred = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="test phase"):

        for i in range(len(batch)):
            batch[i] = batch[i].to(device)

        pred = model(batch)

        all_test_pred.append(pred)

#get pred result
all_test_pred = torch.argmax(torch.cat(all_test_pred, dim=0), dim=-1)

#save result
dataset = pd.read_csv(test_data_path, sep="\t")
dataset["pred"] = all_test_pred.detach().cpu()
dataset.to_csv(os.path.join(save_path, 'test_result_epoch_{}.tsv'.format(epoch)), sep="\t", index=False)