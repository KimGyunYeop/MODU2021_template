from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os


class COLA_dataset(Dataset):
    def __init__(self, file_path, pretrain_model_path):
        super().__init__()
        self.data = pd.read_csv(file_path, sep="\t")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
        print(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return super().__getitem__(index)


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