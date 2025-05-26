import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 10  # TODO: Set the number of epochs

train_val = pd.read_csv("eestec_hackathon_2025_train.tsv", sep='\t')
test_val = train_test_split(train_val, test_size=0.2)
print(test_val)
print(train_val)

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


train_dataset = ReviewDataset(train_val)
test_dataset = ReviewDataset(test_val)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=16, pin_memory=True)

# Additional code if needed
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# TODO: Fill out MyModule
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = UNetBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(64, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.activation = nn.Identity()


def forward(self, x):
    x1 = self.enc1(x)
    x2 = self.enc2(self.pool1(x1))
    x3 = self.bottleneck(self.pool2(x2))

    x = self.up1(x3)
    x = self.dec1(torch.cat([x, x2], dim=1))
    x = self.up2(x)
    x = self.dec2(torch.cat([x, x1], dim=1))

    x = self.final(x)
    return self.activation(x)


model = MyModule().to(DEVICE)

# TODO: Setup loss function, optimiser, and scheduler
criterion = nn.L1Loss
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = None

model.train()
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in tqdm(train_loader, total=len(train_loader)):
        batch = batch.to(DEVICE)

        # TODO: Set up training loop


model.eval()
with torch.no_grad():
    results = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = batch.to(DEVICE)

        # TODO: Set up evaluation loop

    with open("result.txt", "w") as f:
        for val in np.concatenate(results):
            f.write(f"{val}\n")
