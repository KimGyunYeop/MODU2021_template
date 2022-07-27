import torch
import numpy as np
import random
from sklearn.metrics import matthews_corrcoef

def set_seed(seedNum):
    torch.manual_seed(seedNum)
    torch.cuda.manual_seed(seedNum)
    torch.cuda.manual_seed_all(seedNum) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seedNum)
    random.seed(seedNum)

#individual Metric
def MCC(preds, labels):
    assert len(preds) == len(labels)
    return matthews_corrcoef(labels, preds)

def ACC(preds, labels):
    assert len(preds) == len(labels)
    return (preds == labels).mean()

    
def torch_MCC(preds, labels):
    assert len(preds) == len(labels)
    return matthews_corrcoef(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())

def torch_ACC(preds, labels):
    assert len(preds) == len(labels)
    return (preds == labels).detach().cpu().numpy().mean()