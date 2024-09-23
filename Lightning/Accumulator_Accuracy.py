import torch
from torchmetrics import Metric

class Accumulator_Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, success, total):
        self.correct += success
        self.total += total

    def compute(self):
        return self.correct.float() / self.total