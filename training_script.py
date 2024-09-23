import shutil
import subprocess
from Lightning.Callbacks import My_Printing_Callback
from Lightning.Model import NN
from Lightning.Dataset import Data_Module
import pytorch_lightning as pl

from Lightning.Tensor_Dir_Dataset import Tensor_Dir_Dataset
import Lightning.config as cfg
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import torch
import os
from datetime import datetime

if __name__ == "__main__":
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    # Create a folder name based on the model's general hyperparameters
    run_name = f"{current_time}_model-blocks-{cfg.N_SYM_BLOCK}_layers-{cfg.LAYER_PER_BLOCK}_latent-{cfg.LATENT_DIM}"

    # Define the base directory for saving logs and checkpoints
    base_dir = os.path.join("training_runs", run_name)

    # Create the directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    tb_log_dir = os.path.join(base_dir, 'tb_logs')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    profiler_dir = os.path.join(tb_log_dir, 'profiler')

    ## Log config
    # Copy the config file to the base directory
    config_src = os.path.join("Lightning", "config.py")
    config_dst = os.path.join(base_dir, "config.py")
    shutil.copyfile(config_src, config_dst)

    ## Git logging
    # Get the current git commit ID
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    
    # Check if there are uncommitted changes
    git_status = subprocess.check_output(["git", "status", "--porcelain"]).strip().decode()

    # Write the commit ID and changes status to a file
    with open(os.path.join(base_dir, "git_commit_status.txt"), "w") as f:
        f.write(f"Git Commit ID: {commit_id}\n")
        if git_status:
            f.write("Uncommitted changes:\n")
            f.write(git_status + "\n")
        else:
            f.write("No uncommitted changes since the last commit.\n")

    ## Training
    # Initialize network
    logger = TensorBoardLogger(tb_log_dir, name='model_training')
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    )
    
    # Initialize the model
    model = NN(
        alpha=cfg.ALPHA,
        n_sym_block=cfg.N_SYM_BLOCK,
        layer_per_block=cfg.LAYER_PER_BLOCK,
        latent_dim=cfg.LATENT_DIM,
        kernel_size=cfg.KERNEL_SIZE,
        batch_norm_period=cfg.BATCH_NORM_PERIOD
    )
    
    # Initialize the data module
    dm = Tensor_Dir_Dataset(
        batch_size=cfg.BATCH_SIZE,
        val_size=cfg.VAL_SIZE,
        test_size=cfg.TEST_SIZE,
    )

    # Callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",       # Metric to monitor
        dirpath=checkpoint_dir,   # Directory where to save the models
        filename="best_model",    # Filename for the best model
        save_top_k=1,             # Save only the best model
        mode="min"                # Save the model with the minimum val_loss
    )
    
    # Set up the trainer
    trainer = pl.Trainer(
        accelerator=cfg.ACCELERATOR,  # 'gpu' or 'tpu'
        devices=cfg.TRAINER_DEVICES,  # Devices to use
        min_epochs=cfg.MIN_EPOCHS,
        max_epochs=cfg.MAX_EPOCHS,
        precision=cfg.PRECISION,
        # overfit_batches=1, # Debug : Try to overfit the model to one batch
        profiler=profiler,
        # fast_dev_run=True,  # Debug : Smaller loops,
        callbacks=[
            EarlyStopping(monitor="val/loss", patience=cfg.PATIENCE),
            checkpoint_callback,  # Add the checkpoint callback here
        ],
        logger=logger,
    )

    # Start training
    trainer.fit(model, dm)

    # Measurments
    trainer.validate(model, dm)
    trainer.test(model, dm)
