

from typing import Callable

from pytorch_lightning import LightningModule
from torch.nn import Module
from Lightning import config
from Lightning.Model import NN


MODEL_PROVIDER_DICT = {
    "SymCNN_3_16_ELU": lambda:NN(0.95)
}
