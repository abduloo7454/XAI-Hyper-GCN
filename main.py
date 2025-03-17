import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse  # To select model type via command-line arguments
# Import custom modules
import data_loader
from adjacency_matrix import create_trialwise_adjacency_matrices
from graph_creation import create_pyg_graph_data
from dataset_split import split_dataset
from GCN_model import ImprovedGCN
from train import train
from validate import validate
from test import test


# Argument parser for model selection
parser = argparse.ArgumentParser(description="Select Graph Model")
parser.add_argument("--model", type=str, choices=["GCN", "GAT", "GraphSAGE"], default="GCN", help="Choose a model: GCN, GAT, or GraphSAGE")
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define file paths (Update paths accordingly)
data_paths = [
    "Data/Experiment_1/Subject12Trial1.mat",
    "Data/Experiment_1/Subject12Trial2.mat",
    "Data/Experiment_1/Subject34Trial1.mat",
    "Data/Experiment_1/Subject34Trial2.mat",
    "Data/Experiment_2/Subject12Trial1.mat",
    "Data/Experiment_2/Subject12Trial2.mat",
    "Data/Experiment_2/Subject34Trial1.mat",
    "Data/Experiment_2/Subject34Trial2.mat"
]

label_paths = [
    "Data/Experiment_1/Subject12Trial1Label.mat",
    "Data/Experiment_1/Subject12Trial1Label.mat",
    "Data/Experiment_1/Subject12Trial2Label.mat",
    "Data/Experiment_1/Subject12Trial2Label.mat",
    "Data/Experiment_1/Subject34Trial1Label.mat",
    "Data/Experiment_1/Subject34Trial1Label.mat",
    "Data/Experiment_1/Subject34Trial2Label.mat",
    "Data/Experiment_1/Subject34Trial2Label.mat",
    "Data/Experiment_2/Subject12Trial1Label.mat",
    "Data/Experiment_2/Subject12Trial1Label.mat",
    "Data/Experiment_2/Subject12Trial2Label.mat",
    "Data/Experiment_2/Subject12Trial2Label.mat",
    "Data/Experiment_2/Subject34Trial1Label.mat",
    "Data/Experiment_2/Subject34Trial1Label.mat",
    "Data/Experiment_2/Subject34Trial2Label.mat",
    "Data/Experiment_2/Subject34Trial2Label.mat"
]

# Step 1: Load Data
print("Loading EEG data...")
eeg_data = data_loader.load_eeg_data(data_paths)
labels = data_loader.load_labels(label_paths)

# Step 2: Create Adjacency Matrices
print("Creating adjacency matrices...")
adjacency_matrices = create_trialwise_adjacency_matrices(eeg_data, eeg_data)

# Step 3: Convert EEG data to PyTorch Geometric Graphs
print("Creating PyG graph data...")
graph_data_list = create_pyg_graph_data(torch.tensor(eeg_data, dtype=torch.float), adjacency_matrices, torch.LongTensor(labels))

# Step 4: Split Dataset
print("Splitting dataset into train, validation, and test sets...")
train_loader, valid_loader, test_loader = split_dataset(graph_data_list)

# Step 5: Initialize Model, Loss Function, and Optimizer
print("Initializing GCN model...")
num_features = eeg_data.shape[2]  # Number of time points as features
num_classes = 2  # Binary classification (Modify if needed)
# model = ImprovedGCN(num_features=num_features, num_classes=num_classes).to(device)
print(f"Using model: {args.model}")
model = ImprovedGCN(num_features, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Step 6: Training Loop with Early Stopping
print("Starting training process...")

# Metrics Tracking
train_accuracies, val_accuracies, test_accuracies = [], [], []
train_precisions, val_precisions, test_precisions = [], [], []
train_recalls, val_recalls, test_recalls = [], [], []
train_f1s, val_f1s, test_f1s = [], [], []

best_val_loss = float('inf')
stopping_patience = 10
counter = 0
num_epochs = 100

for epoch in range(num_epochs):
    # Train Model
    train_loss, train_acc, train_prec, train_rec, train_f1 = train(train_loader, model, criterion, optimizer, device)

    # Validate Model
    val_loss, val_acc, val_prec, val_rec, val_f1 = validate(valid_loader, model, criterion, device)

    # Test Model
    test_acc, test_prec, test_rec, test_f1, test_cm = test(test_loader, model, device)

    # Store Metrics
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)

    train_precisions.append(train_prec)
    val_precisions.append(val_prec)
    test_precisions.append(test_prec)

    train_recalls.append(train_rec)
    val_recalls.append(val_rec)
    test_recalls.append(test_rec)

    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    test_f1s.append(test_f1)

    # Learning Rate Scheduling
    scheduler.step(val_loss)

    # Print Progress
    print(f"Epoch {epoch+1}: Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Save the best model
        torch.save(model.state_dict(), "best_gcn_model.pth")
    else:
        counter += 1
        if counter >= stopping_patience:
            print("Early stopping triggered.")
            break

# Step 7: Print Final Metrics
print("\nFinal Performance Statistics:")
def print_metrics(title, metrics):
    print(f"{title}: {np.mean(metrics):.4f} ± {np.std(metrics):.4f}")

print_metrics("Average Train Accuracy", train_accuracies)
print_metrics("Average Validation Accuracy", val_accuracies)
print_metrics("Average Test Accuracy", test_accuracies)
print_metrics("Average Train Precision", train_precisions)
print_metrics("Average Validation Precision", val_precisions)
print_metrics("Average Test Precision", test_precisions)
print_metrics("Average Train Recall", train_recalls)
print_metrics("Average Validation Recall", val_recalls)
print_metrics("Average Test Recall", test_recalls)
print_metrics("Average Train F1 Score", train_f1s)
print_metrics("Average Validation F1 Score", val_f1s)
print_metrics("Average Test F1 Score", test_f1s)
