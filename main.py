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

data = pd.read_csv("eestec_hackathon_2025_train.tsv", sep='\t', names=['ID', 'Label', 'Statement', 'Subjects', 'Speaker Name', 'Speaker Title', 'State', 'Party Affiliation', 'Credit History: barely-true', 'Credit History: false', 'Credit History: half-true', 'Credit History: mostly-true', 'Credit History: pants-fire', 'Context/Location'])
data = data.drop(data.columns[0], axis=1)
y = data.iloc[:, 0]

# Remaining columns are input features
X = data.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame):
         #self.id = data_frame['ID']
         #self.label = data_frame['Label']
         self.statement = data_frame['Statement']
         self.subjects = data_frame['Subjects']
         self.speaker_name = data_frame['Speaker Name']
         self.speaker_title = data_frame['Speaker Title']
         self.state = data_frame['State']
         self.party_affiliation = data_frame['Party Affiliation']
         self.credit_history_bt = data_frame['Credit History: barely-true']
         self.credit_history_f = data_frame['Credit History: false']
         self.credit_history_ht = data_frame['Credit History: half-true']
         self.credit_history_mt = data_frame['Credit History: mostly-true']
         self.credit_history_pf = data_frame['Credit History: pants-fire']
         self.cl = data_frame['Context/Location']
    def __len__(self):
         return len(self.statement)

    def __getitem__(self, index):
        #label = self.label.iloc[index]
        statement = self.statement.iloc[index]
        subjects = self.subjects.iloc[index]
        speaker_name = self.speaker_name.iloc[index]
        speaker_title = self.speaker_title.iloc[index]
        state = self.state.iloc[index]
        party_affiliation = self.party_affiliation.iloc[index]
        credit_history_bt = self.credit_history_bt.iloc[index]
        credit_history_f = self.credit_history_f.iloc[index]
        credit_history_ht = self.credit_history_ht.iloc[index]
        credit_history_mt = self.credit_history_mt.iloc[index]
        credit_history_pf = self.credit_history_pf.iloc[index]
        cl = self.cl.iloc[index]
        return statement, subjects, speaker_name, speaker_title, state, party_affiliation,credit_history_bt,credit_history_f,credit_history_ht,credit_history_mt,credit_history_pf,cl


train_dataset = ReviewDataset(X_train)
test_dataset = ReviewDataset(X_test)

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
