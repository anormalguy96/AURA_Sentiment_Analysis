# src/train.py

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from . import config
from . import model
from . import dataset
from . import engine

def run():
    
    df = pd.read_csv(config.DATA_PATH).dropna().sample(n=10000, random_state=42)
    df['sentiment'] = df.Score.apply(lambda score: config.SENTIMENT_MAP.get(score))
    df = df.dropna(subset=['sentiment'])
    df['sentiment'] = df.sentiment.astype(int)

    df_train, df_val = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df.sentiment.values
    )

    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER)

    train_dataset = dataset.ReviewDataset(
        reviews=df_train.Text.to_numpy(),
        targets=df_train.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    val_dataset = dataset.ReviewDataset(
        reviews=df_val.Text.to_numpy(),
        targets=df_val.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device(config.DEVICE)
    aura_model = model.AURA(n_classes=len(config.CLASS_NAMES)).to(device)

    optimizer = AdamW(aura_model.parameters(), lr=config.LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    best_f1 = 0
    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)

        train_acc, train_f1, train_loss = engine.train_epoch(
            aura_model, train_data_loader, loss_fn, optimizer, device, len(df_train)
        )
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f} F1-score {train_f1:.4f}')

        val_acc, val_f1, val_loss = engine.eval_model(
            aura_model, val_data_loader, loss_fn, device, len(df_val)
        )
        print(f'Val loss {val_loss:.4f} accuracy {val_acc:.4f} F1-score {val_f1:.4f}')

        if val_f1 > best_f1:
            torch.save(aura_model.state_dict(), config.SAVED_MODEL_PATH)
            best_f1 = val_f1

if __name__ == '__main__':
    run()