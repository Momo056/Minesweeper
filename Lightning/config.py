import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINER_DEVICES = 'auto'

# Training parameters
BATCH_SIZE = 128
VAL_SIZE = 0.1
TEST_SIZE = 0.1
ACCELERATOR = "gpu"
PRECISION = '16-mixed'
MIN_EPOCHS = 1
MAX_EPOCHS = 1000
PATIENCE = 10

# Hyperparameters for the NN model
ALPHA = 0.95  # Part of loss on the Unknown boundaries
N_SYM_BLOCK = 3
LAYER_PER_BLOCK = 2
LATENT_DIM = 64
KERNEL_SIZE = 3
BATCH_NORM_PERIOD = 2

# Dataset file
TENSOR_DATA_FILE = "dataset/lose_bot/12x12_23253.pt"
