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
from transformers import BertTokenizer, AutoModel

MODEL_NAME = 'distilbert-base-uncased'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label_mapping = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 10  # TODO: Set the number of epochs

data = pd.read_csv("eestec_hackathon_2025_train.tsv", sep='\t', names=['ID', 'Label', 'Statement', 'Subjects', 'Speaker Name', 'Speaker Title', 'State', 'Party Affiliation', 'Credit History: barely-true', 'Credit History: false', 'Credit History: half-true', 'Credit History: mostly-true', 'Credit History: pants-fire', 'Context/Location'])
data = data.drop(data.columns[0], axis=1)
y = data.iloc[:, 0]

# Remaining columns are input features
X = data.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.map(label_mapping).astype(np.float32)
y_test = y_test.map(label_mapping).astype(np.float32)

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, df, labels, tokenizer, max_len=128):
        self.texts = [
            f"{row['Statement']} [SEP] {row['Subjects']} [SEP] "
            f"{row['Speaker Name']}, {row['Speaker Title']} from {row['State']} affiliated with {row['Party Affiliation']} [SEP] "
            f"Context: {row['Context/Location']} [SEP] "
            f"Credit: BT={row['Credit History: barely-true']}, F={row['Credit History: false']}, HT={row['Credit History: half-true']}, "
            f"MT={row['Credit History: mostly-true']}, PF={row['Credit History: pants-fire']}"
            for _, row in df.iterrows()
        ]

        self.encodings = tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

        self.label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.labels = torch.tensor([self.label_map[label] for label in labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['labels'] = self.labels[index]
        return item

import torch
from torch.utils.data import Dataset
import numpy as np # For label processing if needed

# Assuming your label mapping is still relevant for L1Loss
# label_mapping = { "pants-fire": 0.0, "false": 1.0, ..., "true": 5.0 }
# y_train_numeric = y_train.map(label_mapping).astype(np.float32)
# y_test_numeric = y_test.map(label_mapping).astype(np.float32)
# Ensure y_train_numeric and y_test_numeric are available and are pandas Series or NumPy arrays

class TransformerTextDataset(Dataset):
    def __init__(self, texts_series, labels_array, tokenizer, max_len=128):
        self.texts = texts_series.astype(str).tolist() # Ensure texts are strings
        self.labels = labels_array # Should be a NumPy array of numerical labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,    # Max sequence length
            return_token_type_ids=False,# Not needed for DistilBERT/BERT for sentence tasks
            padding='max_length',       # Pad to max_length
            truncation=True,            # Truncate to max_length if longer
            return_attention_mask=True, # Create attention mask
            return_tensors='pt',        # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32) # Ensure label is a tensor
        }


# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True, num_workers=16, pin_memory=True)
# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=BATCH_SIZE,
#                          shuffle=False, num_workers=16, pin_memory=True)
#
train_dataset_hf = TransformerTextDataset(X_train['Statement'], y_train.values, tokenizer, max_len=128)

test_dataset_hf = TransformerTextDataset(X_test['Statement'], y_test.values, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset_hf, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset_hf, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self, model_name, output_dim=1, dropout_rate=0.1):
        super().__init__()
        # Load the pre-trained base model (without a specific head)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # The regressor head
        # self.transformer.config.dim is the hidden size of DistilBERT's [CLS] token embedding
        self.regressor = nn.Linear(self.transformer.config.dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # Pass inputs through the transformer model
        # The output is an object, for DistilBERT, .last_hidden_state is what we need
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get the last hidden state
        last_hidden_state = transformer_output.last_hidden_state
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)

        # For sentence-level tasks, we typically use the representation of the [CLS] token,
        # which is the first token in the sequence.
        cls_token_representation = last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Apply dropout
        pooled_output = self.dropout(cls_token_representation)

        # Pass through the regressor head
        return self.regressor(pooled_output)


# Instantiate the model
model = TransformerRegressor(MODEL_NAME, output_dim=1).to(DEVICE) # output_dim=1 for single value regression


# TODO: Setup loss function, optimiser, and scheduler
criterion = nn.L1Loss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = None

model.train()
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, total=len(train_loader)):
        # Move batch to device
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimiser.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Ensure output shape matches label shape (e.g., [batch_size] vs [batch_size])
        loss = criterion(outputs.squeeze(1), labels) # .squeeze(1) if output_dim is 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimiser.step()
        #scheduler.step() # Update learning rate

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f}")

    # Evaluation
    model.eval()
    total_eval_loss = 0
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(1), labels)

            total_eval_loss += loss.item()
            all_predictions.extend(outputs.squeeze(1).cpu().numpy())

    avg_eval_loss = total_eval_loss / len(test_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Validation Loss: {avg_eval_loss:.4f}")

# Save results
with open("transformer_model_result.txt", "w") as f:
    for val in all_predictions:
        f.write(f"{val}\n")
