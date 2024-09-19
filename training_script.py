from Lightning.Model import NN
from Lightning.Dataset import Data_Module
import pytorch_lightning as pl

import Lightning.config as cfg


# Initialize network
model = NN(cfg.ALPHA)
dm = Data_Module(
    cfg.TENSOR_DATA_FILE,
    cfg.BATCH_SIZE,
    cfg.VAL_SIZE,
)
trainer = pl.Trainer(
    accelerator=cfg.ACCELERATOR,  # 'gpu' or 'tpu'
    devices=cfg.TRAINER_DEVICES,  # Devices to use
    min_epochs=cfg.MIN_EPOCHS,
    max_epochs=cfg.MAX_EPOCHS,
    precision=cfg.PRECISION,
    # overfit_batches=1, # Debug : Try to overfit the model to one batch
    fast_dev_run=True,  # Debug : Smaller loops
)
trainer.fit(model, dm)
trainer.validate(model, dm)
