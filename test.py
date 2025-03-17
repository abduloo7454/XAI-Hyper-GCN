import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix



# Test Function

def test(test_loader, model, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return correct / total, precision_score(all_labels, all_preds, average='macro'), recall_score(all_labels, all_preds, average='macro'), f1_score(all_labels, all_preds, average='macro'), cm
