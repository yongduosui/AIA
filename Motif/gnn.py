import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, GINConv
import torch.nn.functional as F
# from conv_base import GNN_node_Virtualnode
from conv_mol import vGINMolHeadEncoder, GINMolHeadEncoder
import pdb


class GINEncoder(torch.nn.Module):
    
    def __init__(self, num_layer, in_dim, emb_dim):
        super(GINEncoder, self).__init__()

        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout_rate = 0.5
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, 2 * emb_dim),
                                           nn.BatchNorm1d(2 * emb_dim), nn.ReLU(), 
                                           nn.Linear(2 * emb_dim, emb_dim)))
        self.convs = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                                      nn.BatchNorm1d(2 * emb_dim), nn.ReLU(),
                                      nn.Linear(2 * emb_dim, emb_dim)))for _ in range(num_layer - 1)])

    def forward(self, batched_data):
        
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class GINNet(torch.nn.Module):

    def __init__(self, num_class, dataset, num_layer, in_dim=None, emb_dim=300):
        
        super(GINNet, self).__init__()
        self.dataset = dataset
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.num_class = num_class
        if dataset in ["hiv", "pcba"]:
            self.gnn_node = vGINMolHeadEncoder(num_layer, emb_dim)
        else:
            self.gnn_node = GINEncoder(num_layer, in_dim, emb_dim)
        self.pool = global_mean_pool
        self.classifier = torch.nn.Linear(emb_dim, self.num_class)

    def forward(self, batched_data):
        
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.classifier(h_graph)