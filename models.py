
import torch
from torch import nn
from transformers import AutoModel

class COLA_model_baseline(nn.Module):
    def __init__(self, pretrain_model_path) -> None:
        super().__init__()

        #delete this line
        self.tmp = nn.Linear(1,1)
        self.encoder = AutoModel.from_pretrained(pretrain_model_path)
        self.head = nn.Linear(768,2)

    def forward(self, x):
        input_ids = x[0]
        token_type_ids = x[1]
        attention_mask = x[2]

        x = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = self.head(x[1])

        return x


class WiC_model_baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class COPA_model_baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class BoolQ_model_baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


MODEL_DICT = {
    "cola_baseline": COLA_model_baseline,
    "wic_baseline": WiC_model_baseline,
    "copa_baseline": COPA_model_baseline,
    "boolq_baseline": BoolQ_model_baseline,
}