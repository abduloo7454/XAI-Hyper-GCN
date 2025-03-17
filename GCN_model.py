import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class ImprovedGCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ImprovedGCN, self).__init__()
        self.conv1, self.conv2, self.conv3 = GCNConv(num_features, 512), GCNConv(512, 256), GCNConv(256, 128)
        self.bn1, self.bn2, self.bn3 = BatchNorm(512), BatchNorm(256), BatchNorm(128)
        self.dropout, self.out = nn.Dropout(0.6), nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.dropout(torch.relu(self.bn1(self.conv1(x, edge_index))))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(torch.relu(self.bn3(self.conv3(x, edge_index))))
        return self.out(global_mean_pool(x, batch))
