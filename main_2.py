if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    import numpy as np
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
    from torch.optim import AdamW
    from sklearn.model_selection import StratifiedKFold

    NUM_FOLDS = 5
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    NUM_EPOCHS = 3

    # Load and prepare data
    data = pd.read_csv("eestec_hackathon_2025_train.tsv", sep='\t', names=[
        'ID', 'Label', 'Statement', 'Subjects', 'Speaker Name', 'Speaker Title', 'State', 'Party Affiliation',
        'Credit History: barely-true', 'Credit History: false', 'Credit History: half-true',
        'Credit History: mostly-true', 'Credit History: pants-fire', 'Context/Location'
    ])
    data = data.drop(data.columns[0], axis=1)
    y = data.iloc[:2000, 0]
    X = data.iloc[:2000, 1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    class ReviewDataset(Dataset):
        def __init__(self, df, labels, tokenizer, max_len=128):
            self.texts = [
                f"{row['Statement']} [SEP] {row['Subjects']} [SEP] "
                f"{row['Speaker Name']}, {row['Speaker Title']} from {row['State']} affiliated with {row['Party Affiliation']} [SEP] "
                f"Context: {row['Context/Location']} [SEP] "
                f"Credit: BT={row['Credit History: barely-true']}, F={row['Credit History: false']}, "
                f"HT={row['Credit History: half-true']}, MT={row['Credit History: mostly-true']}, "
                f"PF={row['Credit History: pants-fire']}"
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

    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(set(y))
    ).to(DEVICE)


    patience = 5
    best_val_loss = float('inf')
    best_val_loss_fold = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    criterion = nn.CrossEntropyLoss()

    model.train()


    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_dataset = ReviewDataset(X_train, y_train, tokenizer)
        val_dataset = ReviewDataset(X_val, y_val, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(set(y)),
            hidden_dropout_prob=0.1  # Increase dropout
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Add weight decay
        total_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps

        )
        best_val_loss = float('inf')

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_train_loss = 0
            epochs_without_improvement = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()
            avg_val_loss = total_val_loss / len(test_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break


        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        if best_val_loss < best_val_loss_fold:
            best_val_loss_fold = best_val_loss
            best_model_state = model.state_dict()


    if best_model_state:
        print(f"Best validation loss was {best_val_loss_fold}")
        model.load_state_dict(best_model_state)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    with open("result.txt", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
