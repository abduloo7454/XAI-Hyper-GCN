import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# Validation Function

def validate(valid_loader, model, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    return total_loss / len(valid_loader), correct / total, precision_score(all_labels, all_preds, average='macro'), recall_score(all_labels, all_preds, average='macro'), f1_score(all_labels, all_preds, average='macro')
