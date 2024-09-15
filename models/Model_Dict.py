from typing import Callable
from torch.nn import Module
import torch.nn as nn
from models.Symetry_Invariant_Conv2D import Symetry_Inveriant_Conv2D

MODEL_PROVIDER_DICT : dict[str, Callable[[], Module]] = {
    'SymCNN_3_16_ELU' : lambda: nn.Sequential(
        Symetry_Inveriant_Conv2D(
            nn.Sequential(
                nn.Conv2d(10, 16, 9, padding='same', padding_mode='zeros'),
                nn.ELU(),
            )
        ),
        Symetry_Inveriant_Conv2D(
            nn.Sequential(
                nn.Conv2d(16, 16, 9, padding='same', padding_mode='zeros'),
                nn.ELU(),
            )
        ),
        Symetry_Inveriant_Conv2D(
            nn.Sequential(
                nn.Conv2d(16, 2, 1, padding='same', padding_mode='zeros'),
                nn.ELU(),
            )
        ),
        nn.LogSoftmax(-3),
    )
}