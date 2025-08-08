import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    all_targets = []
    all_predictions = []

    for d in tqdm(data_loader, total=len(data_loader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        token_type_ids = d["token_type_ids"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            ids=input_ids,
            mask=attention_mask,
            token_type_ids=token_type_ids
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    f1 = f1_score(all_targets, all_predictions, average='weighted')
    return correct_predictions.double() / n_examples, f1, sum(losses) / len(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                ids=input_ids,
                mask=attention_mask,
                token_type_ids=token_type_ids
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    return correct_predictions.double() / n_examples, f1, sum(losses) / len(losses)