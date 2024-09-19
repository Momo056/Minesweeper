import torch

# Set device cuda for GPU if it's available otherwise run on the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINER_DEVICES = [0]

# Hyperparameters
BATCH_SIZE = 128
ALPHA = 0.95  # Part of loss on the Unknown boundaries
VAL_SIZE = 0.1
TENSOR_DATA_FILE = "dataset/lose_bot/12x12_23253.pt"
ACCELERATOR = "gpu"
PRECISION = 16
MIN_EPOCHS = 1
MAX_EPOCHS = 5
