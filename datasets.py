import token
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os
import torch


class COLA_dataset(Dataset):
    def __init__(self, file_path, pretrain_model_path, max_len):
        super().__init__()
        self.data = pd.read_csv(file_path, sep="\t")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
        self.max_len = max_len
        print(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentData = self.data.iloc[index,:]
            
        tok = self.tokenizer(sentData['sentence'], padding="max_length", max_length=self.max_len, truncation=True)

        input_ids=torch.LongTensor(tok["input_ids"])
        token_type_ids=torch.LongTensor(tok["token_type_ids"])
        attention_mask=torch.LongTensor(tok["attention_mask"])
        
        label = None
        if 'acceptability_label' in sentData.keys():
            label = sentData['acceptability_label']
            
        return input_ids, token_type_ids, attention_mask, label


class WiC_dataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        return super().__getitem__(index)


class COPA_dataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        return super().__getitem__(index)


class BoolQ_dataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        return super().__getitem__(index)


DATASET_DICT = {
    "cola_baseline": COLA_dataset,
    "wic_baseline": WiC_dataset,
    "copa_baseline": COPA_dataset,
    "boolq_baseline": BoolQ_dataset,
}