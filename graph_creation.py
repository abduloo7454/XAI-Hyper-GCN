import torch
from torch_geometric.data import Data
import numpy as np

def create_pyg_graph_data(eeg_data, adjacency_matrices, labels):
    """
    Convert EEG data and adjacency matrices into PyTorch Geometric graph data objects.
    """
    graph_data_list = []

    # Ensure tensor copying is done properly
    eeg_data = eeg_data.clone().detach()
    labels = labels.clone().detach()

    # Debugging prints
    print(f"eeg_data shape: {eeg_data.shape}")  # Should be (num_samples, num_channels, num_time_points)
    print(f"labels shape: {labels.shape}")      # Should be (num_samples,)

    for i in range(eeg_data.shape[0]):  # Ensure using the right index
        x = eeg_data[i]  # Shape: (channels, time_points)
        edge_index = np.vstack(np.nonzero(adjacency_matrices[i])).astype(np.int64)
        
        if i >= labels.shape[0]:  # Check if index exists
            print(f"Skipping index {i} - label index out of range")
            continue

        y = labels[i]  # Ensure label exists

        # Convert to PyTorch Geometric Data object
        data = Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long), y=y)
        graph_data_list.append(data)

    return graph_data_list
