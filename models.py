
import torch
from torch import nn
from transformers import AutoModel

class COLA_model_baseline(nn.Module):
    def __init__(self, pretrain_model_path) -> None:
        super().__init__()

        #delete this line
        self.tmp = nn.Linear(1,1)

    def forward(self, x):

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