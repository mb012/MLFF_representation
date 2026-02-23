

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch.nn import LayerNorm


class GNN(torch.nn.Module):
    def __init__(self, in_dim, dim, out_dim, num_layers=10, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_dim, dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(dim, dim))
        self.convs.append(GCNConv(dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class ImprovedGNN(torch.nn.Module):
    def __init__(self, in_dim, dim, out_dim, num_layers=10, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()  
        self.convs.append(GCNConv(in_dim, dim))
        self.norms.append(LayerNorm(dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(dim, dim))
            self.norms.append(LayerNorm(dim))
        
        self.convs.append(GCNConv(dim, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.convs[:-1], self.norms)):
            x = conv(x, edge_index)
            x = norm(x)  
            x = F.silu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)  
        return x
    
