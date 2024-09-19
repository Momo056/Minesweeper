from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class My_Printing_Callback(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("Training start !")
        return super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("Training end !")
        return super().on_train_end(trainer, pl_module)
