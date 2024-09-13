
# Test loss
lose_bot/12x12_23253.pt
random_seed = 86431
n_epoch = 50
batch_size = 64
random_seed = 86431
train_data, test_data, train_mines, test_mines = train_test_split(*dataset, test_size=0.1, shuffle=False, random_state=random_seed) # Do not shuffle to not mix the same grids in training and test
train_dataset = TensorDataset(train_data.type(torch.float32).to(device), train_mines)
test_dataset = TensorDataset(test_data.type(torch.float32).to(device), test_mines)

model = nn.Sequential(
    nn.Conv2d(10, 16, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(16, 16, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(16, 2, 1),
    nn.LogSoftmax(0),
)
548.4662997769776


model = nn.Sequential(
    nn.Conv2d(10, 16, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(16, 2, 1),
    nn.LogSoftmax(0),
)
548.9796941335716

model = nn.Sequential(
    nn.Conv2d(10, 2, 5, padding='same', padding_mode='zeros'),
    nn.LogSoftmax(0),
)
555.9844237021711

model = nn.Sequential(
    nn.Conv2d(10, 16, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(32, 32, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(32, 16, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(16, 2, 1),
    nn.LogSoftmax(0),
)
548.4208509948678

model = nn.Sequential(
    nn.Conv2d(10, 64, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(64, 64, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(64, 32, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(32, 32, 5, padding='same', padding_mode='zeros'),
    nn.ReLU(),
    nn.Conv2d(32, 2, 1),
    nn.LogSoftmax(0),
)
548.6172647517197

