import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def train(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    return total_loss / len(train_loader), correct / total, precision_score(all_labels, all_preds, average='macro'), recall_score(all_labels, all_preds, average='macro'), f1_score(all_labels, all_preds, average='macro')