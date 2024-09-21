from Lightning.Callbacks import My_Printing_Callback
from Lightning.Model import NN
from Lightning.Dataset import Data_Module
import pytorch_lightning as pl

import Lightning.config as cfg
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import torch

if __name__ == "__main__":
    # Initialize network
    logger = TensorBoardLogger('tb_logs', name='tutorial')
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('tb_logs/tutorial_profiler'),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    # )
    profiler = None
    model = NN(cfg.ALPHA)
    dm = Data_Module(
        cfg.TENSOR_DATA_FILE,
        cfg.BATCH_SIZE,
        cfg.VAL_SIZE,
    )
    
    # Callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       # Metric to monitor
        dirpath="checkpoints/",   # Directory where to save the models
        filename="best_model",    # Filename for the best model
        save_top_k=1,             # Save only the best model
        mode="min"                # Save the model with the minimum val_loss
    )
    
    trainer = pl.Trainer(
        profiler=profiler,
        accelerator=cfg.ACCELERATOR,  # 'gpu' or 'tpu'
        devices=cfg.TRAINER_DEVICES,  # Devices to use
        min_epochs=cfg.MIN_EPOCHS,
        max_epochs=cfg.MAX_EPOCHS,
        precision=cfg.PRECISION,
        # overfit_batches=1, # Debug : Try to overfit the model to one batch
        # fast_dev_run=True,  # Debug : Smaller loops,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE),
            checkpoint_callback,  # Add the checkpoint callback here
            My_Printing_Callback(),
        ],
        logger=logger,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
