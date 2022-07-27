from turtle import forward
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from conifg import DATAPATH_DICT
from models import MODEL_DICT
from datasets import DATASET_DICT

assert MODEL_DICT.keys() == DATASET_DICT.keys(), "Model and dataset not matched"

#set config and hyperparameter
task = "cola"
model_name = "cola_baseline"
pretrain_model_path = "monologg/kobert"
save_path = "result/tmp"
gpu = "CUDA:0"
batch_size = "128"
max_length = "100"
lr = "2e-5"
eps = "1e-8"

train_data_path = DATAPATH_DICT[task]["train"]
val_data_path = DATAPATH_DICT[task]["val"]

#load dataset and dataloader
train_dataset = DATASET_DICT[model_name](train_data_path, pretrain_model_path)
dev_dataset = DATASET_DICT[model_name](val_data_path, pretrain_model_path)

#load model
model = MODEL_DICT[model_name]()

#set criterion
lf = CrossEntropyLoss()

#set optimizer
optimizer = Adam(model.parameters(), lr=lr, eps=eps)

