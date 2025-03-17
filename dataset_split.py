import random
from torch_geometric.data import DataLoader

def split_dataset(graph_data_list, train_ratio=0.75, valid_ratio=0.15):
    """
    Shuffle and split the dataset into train, validation, and test sets.
    """
    random.shuffle(graph_data_list)
    num_samples = len(graph_data_list)
    train_index, valid_index = round(train_ratio * num_samples), round((train_ratio + valid_ratio) * num_samples)
    
    train_set, valid_set, test_set = graph_data_list[:train_index], graph_data_list[train_index:valid_index], graph_data_list[valid_index:]
    
    return (DataLoader(train_set, batch_size=8, shuffle=True),
            DataLoader(valid_set, batch_size=8, shuffle=False),
            DataLoader(test_set, batch_size=8, shuffle=False))
