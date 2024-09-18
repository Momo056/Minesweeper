

from typing import Any
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset


from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

class Abstract_Regression_Trainer:
    def __init__(self, 
                 model: Module, 
                 optimizer: Optimizer, 
                 batch_size: int = 32, 
                 device: str = 'cuda',
                 training_epoch: int = 10,
                 ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.training_epoch = training_epoch
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None

    def fit(self, train_data: Dataset, test_data: Dataset):
        self.train_loader = DataLoader(train_data, self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, self.batch_size, shuffle=True)
        for i in range(self.training_epoch):
            self.train_epoch()
            self.test_epoch()

            if self.early_stoping():
                return
            
    def entry_to_device(self, entry: Any):
        return entry.to(self.device)
    
    def change_device_loader(self, loader: DataLoader)!
        for entry in loader:
            yield self.entry_to_device(entry)

    def train_epoch(self):
        for self.entry in self.change_device_loader(self.train_loader):
            self.optimizer.zero_grad()

            loss = self.comput_loss()

            loss.backward()
            self.optimizer.step()

            self.train
    
    def comput_loss(self)


